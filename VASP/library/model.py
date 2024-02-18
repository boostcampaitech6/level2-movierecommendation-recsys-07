import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, args):
        super(MultiDAE, self).__init__()
        p_dims = args.model.p_dims
        q_dims = args.model.q_dims
        dropout = args.model.dropout
        self.p_dims = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
            ]
        )
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, args):
        super(MultiVAE, self).__init__()
        p_dims = args.model.p_dims + [args.n_items]
        q_dims = args.model.q_dims
        dropout = args.model.dropout
        self.p_dims = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
            ]
        )
        self.p_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
            ]
        )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class VASP(MultiVAE):
    def __init__(self, args):
        super().__init__(args)
        self.weight = nn.Parameter(torch.randn(args.n_items, args.n_items))
        self.register_buffer(
            "mask", torch.ones(args.n_items, args.n_items) - torch.eye(args.n_items)
        )

    def nease(self, input):
        weight = self.weight * self.mask
        linear_output = F.linear(input, weight)
        return linear_output

    def vae(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward(self, input):
        vae_result, mu, logvar = self.vae(input)
        nease_result = torch.sigmoid(self.nease(input))
        hadamard_product_result = vae_result * nease_result
        return hadamard_product_result, mu, logvar
