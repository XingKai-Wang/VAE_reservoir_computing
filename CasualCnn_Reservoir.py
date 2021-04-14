import torch
import torch.nn as nn
from CasualConv1d import CasualConv1d

class CasualReservoirEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, z_dim,  *args):
        super(CasualReservoirEncoder, self).__init__()
        self.num_filters = num_filters
        self.z_dim = z_dim
        hidden_filters = num_filters

        self.Conv1d = nn.Sequential(
            CasualConv1d(in_channels,out_channels,kernel_size=(3,),dilation=1,A=False),
            nn.LeakyReLU()
        )
        for p in self.parameters():
            p.requires_grad = False

        self.encoder = nn.Sequential(
            nn.Conv2d(out_channels, self.num_filters, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)), # 28x28 -> 14x14
            nn.LeakyReLU(),
            nn.Conv2d(hidden_filters, 2 * hidden_filters, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)), # 14x14 -> 7x7
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2*hidden_filters*7*7, self.z_dim)

        )

    def forward(self, x):
        x = self.Conv1d(x.view(x.shape[0], 1, 784))
        h_e = self.encoder(x.view(x.shape[0], -1, 28, 28))
        mu, log_var = torch.chunk(h_e, 2, dim=1)

        return mu, log_var

class CasualReservoirDecoder(nn.Module):
    def __init__(self, z_dim, out_channels, num_filters, **kwargs):
        super(CasualReservoirDecoder, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        hidden_filters = num_filters

        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, 2*hidden_filters*7*7),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*hidden_filters, hidden_filters, kernel_size=(4,), padding=(1,), stride=(2,)), #7x7 -> 14x14
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_filters, out_channels, kernel_size=(4,), padding=(1,), stride=(2,)), # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 7, 7)
        x_recon = self.decoder(x)

        return x_recon
