import torch
import torch.nn as nn
from CausalConv1d import CausalConv1d

class CausalReservoirEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, z_dim,  *args):
        super(CausalReservoirEncoder, self).__init__()
        self.num_filters = num_filters
        self.z_dim = z_dim
        hidden_filters = num_filters

        self.conv1d = nn.Sequential(
            CausalConv1d(in_channels,out_channels,kernel_size=3,dilation=1,A=False),
            nn.LeakyReLU(),
            CausalConv1d(out_channels,out_channels,kernel_size=3,dilation=1,A=False),
            nn.LeakyReLU(),
            CausalConv1d(out_channels, out_channels,kernel_size=3,dilation=2,A=False),
            nn.LeakyReLU(),
            CausalConv1d(out_channels,out_channels,kernel_size=3,dilation=4,A=False),
            nn.LeakyReLU()
        )
        for p in self.parameters():
            p.requires_grad = False

        self.encoder = nn.Sequential(
            nn.Conv2d(out_channels, self.num_filters, kernel_size=4, padding=1, stride=2), # 28x28 -> 14x14
            nn.LeakyReLU(),
            nn.Conv2d(hidden_filters, 2 * hidden_filters, kernel_size=4, padding=1, stride=2), # 14x14 -> 7x7
            nn.LeakyReLU(),
            nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size=3, padding=1, stride=1),  # 7x7 -> 7x7
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4*hidden_filters*7*7, self.z_dim)

        )

    def forward(self, x):
        x = self.conv1d(x.view(x.shape[0], 1, 784))
        h_e = self.encoder(x.view(x.shape[0], -1, 28, 28))
        mu, log_var = torch.chunk(h_e, 2, dim=1)

        return mu, log_var

class CausalReservoirDecoder(nn.Module):
    def __init__(self, z_dim, out_channels, num_filters, **kwargs):
        super(CausalReservoirDecoder, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        hidden_filters = num_filters

        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, 4*hidden_filters*7*7),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            # output=(input - 1) * stride + output_padding - 2 * padding + kernel
            nn.ConvTranspose2d(4 * hidden_filters, 2 * hidden_filters, kernel_size=3, padding=1, output_padding=0,stride=1),  # 7x7 -> 7x7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*hidden_filters, hidden_filters, kernel_size=4, padding=1, stride=2), #7x7 -> 14x14
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_filters, out_channels, kernel_size=4, padding=1, stride=2), # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 7, 7)
        x_recon = self.decoder(x)

        return x_recon
