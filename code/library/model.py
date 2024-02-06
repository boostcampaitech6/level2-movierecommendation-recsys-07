import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        try:  # visualize를 위해 embedding을 꺼낼 때 args.n_users가 없음.
            self.user_dims = args.n_users
            self.item_dims = args.n_items
        except:
            self.user_dims = 31360
            self.item_dims = 6807
        # hidden_dim을 embeddding dim으로 사용
        self.user_embedding = nn.Embedding(self.user_dims, args.model.hidden_dim)
        self.item_embedding = nn.Embedding(self.item_dims, args.model.hidden_dim)
        torch.nn.init.xavier_normal_(self.user_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.item_embedding.weight.data)
        # 전체 정답률 평균
        try:  # visualizae를 위해 embedding을 꺼낼 때 args.n_rows가 없음.
            self.mu = args.n_rows / (args.n_rows + args.n_users * args.dataloader.n_neg)
        except:
            pass
        self.b_u = nn.Parameter(torch.zeros(self.user_dims))
        self.b_i = nn.Parameter(torch.zeros(self.item_dims))

    def forward(self, x):
        uid = x[:, 0]
        iid = x[:, 1]

        user_x = self.user_embedding(uid)
        item_x = self.item_embedding(iid)
        dot = (user_x * item_x).sum(dim=1)
        return self.mu + dot + self.b_u[uid] + self.b_i[iid]


class LMF(MF):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, x):
        uid = x[:, 0]
        iid = x[:, 1]

        user_x = self.user_embedding(uid)
        item_x = self.item_embedding(iid)
        dot = (user_x * item_x).sum(dim=1)
        logit = self.mu + dot + self.b_u[uid] + self.b_i[iid]
        return torch.exp(logit) / (1 + torch.exp(logit))


class FM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feat_dim = args.feat_dim

        # bias and first order interaction embedding
        # user, item은 one-hot encoding이 아니므로 Embedding으로 구현
        # features는 multi-hot encoding이므로 Linear로 구현
        self.bias = nn.Parameter(torch.zeros(1))
        self.first_user_emb = nn.Embedding(args.n_users, 1)
        self.first_item_emb = nn.Embedding(args.n_items, 1)
        self.first_feat_emb = nn.ModuleList(
            [nn.Linear(self.feat_dim[i], 1) for i in range(len(self.feat_dim))]
        )

        # second order interaction embedding
        self.second_user_emb = nn.Embedding(args.n_users, args.model.hidden_dim)
        self.second_item_emb = nn.Embedding(args.n_items, args.model.hidden_dim)
        self.second_feat_emb = nn.ModuleList(
            [
                nn.Linear(self.feat_dim[i], args.model.hidden_dim)
                for i in range(len(self.feat_dim))
            ]
        )

        # Initialize
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0)

    def forward(self, x):
        # user와 item은 이진수 표현이 아님.
        user_x = self.first_user_emb(x[:, 0])
        item_x = self.first_item_emb(x[:, 1])

        # 나머지 feature는 이진수 표현.
        # multi-hot encoding 후 normalize. 하나의 feature에 여러 embedding이 합쳐져 커지는 것을 방지하기 위함.
        feat_x = torch.concat(
            [
                nn.functional.normalize(
                    self.first_feat_emb[i](
                        self.binary(x[:, i + 2], self.args.feat_dim[i])
                    )
                )
                for i in range(len(self.feat_dim))
            ],
            dim=1,
        )
        # first_x = (batch, 1)
        first_x = torch.sum(
            torch.concat([user_x, item_x, feat_x], dim=1), dim=1, keepdim=True
        )

        # for memory
        del user_x, item_x, feat_x

        ## second order interaction
        # first order와 비슷한 방식으로 embedding
        # square_of_sum sum_of_square 계산을 위해 unsqueeze.
        # userx2, itemx2 = (batch, 1, hidden_dim)
        # featx2 = (batch, n_feat, hidden_dim)
        user_x2 = self.second_user_emb(x[:, 0]).unsqueeze(1)
        item_x2 = self.second_item_emb(x[:, 1]).unsqueeze(1)
        feat_x2 = torch.concat(
            [
                nn.functional.normalize(
                    self.second_feat_emb[i](
                        self.binary(x[:, i + 2], self.args.feat_dim[i])
                    )
                ).unsqueeze(1)
                for i in range(len(self.feat_dim))
            ],
            dim=1,
        )

        # second_emb = (batch, 2 + n_feat, hidden_dim)
        second_emb = torch.concat([user_x2, item_x2, feat_x2], dim=1)

        # for memory
        del user_x2, item_x2, feat_x2

        # square_of_sum, sum_of_square = (batch, hidden_dim)
        square_of_sum = torch.sum(second_emb, dim=1) ** 2
        sum_of_square = torch.sum(second_emb**2, dim=1)
        # second_x = (batch, 1)
        second_x = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return (self.bias + first_x + second_x).squeeze(1)

    def binary(self, x: torch.tensor, bits: int):
        """
        이진수 x를 bits 차원으로 multi-hot encoding하는 함수.
        binary(torch.tensor([[9],[7]]), 5)
        = torch.tensor([[1., 0., 0., 1., 0.], [1., 1., 1., 0., 0.]])
        """
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float().squeeze()
