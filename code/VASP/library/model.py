import torch
import torch.nn as nn
import torch.nn.functional as F


class NEASE(nn.Module):
    def __init__(self, args):
        super(NEASE, self).__init__()
        n_items = args.n_items
        self.B = torch.nn.Parameter(torch.Tensor(n_items, n_items))
        # 파라미터 초기화, 대각성분을 제외하고 학습이 가능하도록
        self.weight = nn.Parameter(torch.randn(n_items, n_items))
        # 대각성분을 0으로 설정하는 마스크 생성
        self.register_buffer("mask", torch.ones(n_items, n_items) - torch.eye(n_items))

    def forward(self, x):
        # 대각성분이 0인 가중치 생성
        weight = self.weight * self.mask
        # 선형 변환 적용
        linear_output = F.linear(x, weight)
        # sigmoid 활성화 함수 적용
        return linear_output
