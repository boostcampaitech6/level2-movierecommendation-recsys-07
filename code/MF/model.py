import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, args):
        super().__init__()
        try:  # visualize를 위해 embedding을 꺼낼 때 args.n_users가 없음.
            self.user_dims = args.n_users
            self.item_dims = args.n_items
        except:
            self.user_dims = 31360
            self.item_dims = 6807
        # hidden_dim을 embeddding dim으로 사용
        self.user_embedding = nn.Embedding(self.user_dims, args.hidden_dim)
        self.item_embedding = nn.Embedding(self.item_dims, args.hidden_dim)
        torch.nn.init.xavier_normal_(self.user_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.item_embedding.weight.data)
        # 전체 정답률 평균
        try:  # visualizae를 위해 embedding을 꺼낼 때 args.n_rows가 없음.
            self.mu = args.n_rows / (args.n_rows + args.n_users * args.n_neg)
        except:
            pass
        self.b_u = nn.Parameter(torch.zeros(self.user_dims))
        self.b_i = nn.Parameter(torch.zeros(self.item_dims))

    def forward(self, x):
        uid = x[:, 0].to(torch.int64)
        iid = x[:, 1].to(torch.int64)

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
