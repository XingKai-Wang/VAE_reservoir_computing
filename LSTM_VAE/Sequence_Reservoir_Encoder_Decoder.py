import torch
import torch.nn as nn

class ReserviorEncoder(nn.Module):
    def __init__(self, input_dim, reservoir_dim, input_channels, num_filters, z_dim):
        super(ReserviorEncoder, self).__init__()
        self.input_dim = input_dim
        self.r_dim = reservoir_dim
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.z_dim = z_dim

        self.Win = torch.randn(self.input_dim, self.r_dim).cuda()
        self.Wr = torch.randn(self.r_dim, self.r_dim).cuda()
        self.U = torch.randn(self.input_dim, self.r_dim).cuda()

        self.convs = nn.Sequential(
            # output = ([(input + 2 * padding - dilation * (kernel - 1) - 1 ]/ stride) + 1
            nn.Conv3d(self.input_channels, self.num_filters, padding=1, kernel_size=(3,3,3), stride=(1,2,2)),  # 32x32 -> 16x16
            nn.LeakyReLU(),
            nn.Conv3d(self.num_filters, 2 * self.num_filters, padding=(1,0,0),kernel_size=(3,4,4), stride=(1,2,2)), # 16x16 -> 7x7
            nn.LeakyReLU())

        self.linear = nn.Linear(2 * self.num_filters * 7 * 7, self.z_dim)

    def encode(self, input):
        H = []
        for seq_index in range(input.size(1)):
            x = input[:,seq_index,:,:]
            # initialize reservoir
            # h = tanh(x * Win)
            if seq_index == 0:
                h = torch.tanh(torch.matmul(x.view(x.size(0), -1, 64 * 64), self.Win))
            # hi = tanh(x * U + hi-1 * Wr)
            else:
                h = torch.tanh(torch.matmul(x.view(x.size(0), -1, 64 * 64), self.U) + torch.matmul(h, self.Wr))

            H.append(h)
        return torch.stack(H, dim=1)

    def forward(self, x):
        h = self.encode(x)
        h = h.reshape(x.size(0), 20, -1, 32, 32).permute(0, 2, 1, 3, 4)
        h_e = self.convs(h)
        h_e = self.linear(h_e.view(x.size(0), -1, 20, 2*self.num_filters*7*7))

        mu, log_var = torch.chunk(h_e, 2, dim=3)

        return mu, log_var

class ReservoirDecoder(nn.Module):
    def  __init__(self, z_dim, num_filters, output_channels):
        super(ReservoirDecoder, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        self.output_channels = output_channels

        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, 2 * self.num_filters * 7 * 7),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            # output = (input − 1) × stride − 2 × padding + dilation × (kernel − 1) + output_padding + 1
            nn.ConvTranspose3d(2 * self.num_filters, self.num_filters, kernel_size=(3,4,4), padding=(1,0,0), output_padding=0, stride=(1,2,2)), # 7x7 -> 16x16
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.num_filters, self.num_filters, kernel_size=3, padding=1, output_padding=(0,1,1),stride=(1,2,2)),  # 16x16 -> 32x32
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.num_filters, self.output_channels, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2)) # 32x32 -> 64x64
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.reshape([x.size(0), -1, 20, 7, 7])
        x_recon = torch.sigmoid(self.decoder(x)).permute(0, 2, 1, 3, 4)

        return x_recon

