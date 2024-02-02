import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.user_dims = args.n_users
        self.item_dims = args.n_items
        # hidden_dim을 embeddding dim으로 사용
        self.user_embedding = nn.Embedding(self.user_dims, args.hidden_dim)
        self.item_embedding = nn.Embedding(self.item_dims, args.hidden_dim)
        torch.nn.init.xavier_normal_(self.user_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.item_embedding.weight.data)
        # 전체 정답률 평균
        self.mu = args.n_rows / (args.n_rows + args.n_users * args.n_neg)
        self.b_u = nn.Parameter(torch.zeros(self.user_dims))
        self.b_i = nn.Parameter(torch.zeros(self.item_dims))

    def forward(self, x):
        uid = x[:, 0].to(torch.int64)
        iid = x[:, 1].to(torch.int64)

        user_x = self.user_embedding(uid)
        item_x = self.item_embedding(iid)
        dot = (user_x * item_x).sum(dim=1)
        return self.mu + dot + self.b_u[uid] + self.b_i[iid]
