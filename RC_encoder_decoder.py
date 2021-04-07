import torch
import torch.nn as nn

class RCEncoder(nn.Module):
    def __init__(self, input_dim, reservoir_dim, z_dim, T):
        '''

        :param input_dim: input dimension of x
        :param reservoir_dim: dimension of adjacency matrix in reservoir/the number of neurons in reservoir layer
        :param output_dim: output dimension
        :param T: recurrent time
        '''
        super(RCEncoder, self).__init__()
        self.in_dim = input_dim
        self.r_dim = reservoir_dim
        self.z_dim = z_dim
        self.Win = torch.randn(self.in_dim, self.r_dim)
        self.Wr = torch.randn(self.r_dim, self.r_dim)
        self.U = torch.randn(self.in_dim, self.r_dim)
        self.T = T

        self.linear = nn.Sequential(
            nn.Linear(self.r_dim, self.r_dim),
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.z_dim)
        )

    def encode(self, x):
        for t in range(self.T):
            # initialize reservoir
            # h = tanh(x * Win)
            if t == 0 :
                h = torch.tanh(torch.matmul(x, self.Win))
            # hi = tanh(x * U + hi-1 * Wr)
            else:
                h = torch.tanh(torch.matmul(x, self.U) + torch.matmul(h, self.Wr))

        return h

    def forward(self, h):
        h_e = self.linear(h)
        mu, log_var = torch.chunk(h_e, 2, dim=1)
        return mu, log_var


class RCDecoder(nn.Module):
    def __init__(self, z_dim, reservoir_dim, output_dim):
        '''

        :param z_dim: laten dimension
        :param reservoir_dim: dimension of adjacency matrix in reservoir/the number of neurons in reservoir layer
        :param output_dim: reconstruct data with 784 dimensions (28 * 28)
        '''
        super(RCDecoder, self).__init__()
        self.z_dim = z_dim
        self.r_dim = reservoir_dim
        self.out_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.r_dim),
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.r_dim),
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.out_dim)
        )

    def forward(self, z):
        x_mean = self.decoder(z)
        x_recon = torch.sigmoid(x_mean)

        return x_recon
