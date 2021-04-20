import torch
import torch.nn as nn

class RCEncoder(nn.Module):
    def __init__(self, input_dim, reservoir_dim, num_input_channels, num_filters, z_dim, T, layer_type):
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
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.Win = torch.randn(self.in_dim, self.r_dim).cuda()
        self.Wr = torch.randn(self.r_dim, self.r_dim).cuda()
        self.U = torch.randn(self.in_dim, self.r_dim).cuda()
        self.T = T
        self.layer_type = layer_type
        hidden_filters = num_filters

        self.linear = nn.Sequential(
            nn.Linear(self.r_dim, self.r_dim),
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.z_dim)
        )

        self.convs = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_filters, kernel_size=4,stride=2), # 16x16 -> 7x7
            nn.LeakyReLU(),
            # nn.Conv2d(hidden_filters, 2 * hidden_filters, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),  # 7x7 -> 4x4
            # nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(hidden_filters * 7 * 7, z_dim)
        )

    def encode(self, x):
        for t in range(self.T):
            # initialize reservoir
            # h = tanh(x * Win)
            if t == 0 :
                h = torch.tanh(torch.matmul(x.view(-1, 784), self.Win))
            # hi = tanh(x * U + hi-1 * Wr)
            else:
                h = torch.tanh(torch.matmul(x.view(-1, 784), self.U) + torch.matmul(h, self.Wr))

        return h

    def forward(self, x):
        h = self.encode(x)
        if self.layer_type == 'mlp':
            h_e = self.linear(h)
        elif self.layer_type == 'cnn':
            h = h.reshape(h.shape[0], 1, 16, 16)
            h_e = self.convs(h)
        mu, log_var = torch.chunk(h_e, 2, dim=1)
        return mu, log_var


class RCDecoder(nn.Module):
    def __init__(self, z_dim, reservoir_dim, output_dim, num_output_channels, num_filters, layer_type):
        '''

        :param z_dim: laten dimension
        :param reservoir_dim: dimension of adjacency matrix in reservoir/the number of neurons in reservoir layer
        :param output_dim: reconstruct data with 784 dimensions (28 * 28)
        '''
        super(RCDecoder, self).__init__()
        self.z_dim = z_dim
        self.r_dim = reservoir_dim
        self.out_dim = output_dim
        self.num_output_channels = num_output_channels
        self.num_filters = num_filters
        self.layer_type = layer_type

        self.linear_decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.r_dim),
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.r_dim),
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.out_dim[1] * self.out_dim[2])
        )

        hidden_filters = num_filters
        self.linear = nn.Sequential(
            nn.Linear(z_dim, hidden_filters * 7 * 7),
            nn.LeakyReLU()
        )
        self.convs_decoder = nn.Sequential(
            # output=(input - 1) * stride + output_padding - 2 * padding + kernel
            # nn.ConvTranspose2d(2 * hidden_filters, hidden_filters, kernel_size=(3, 3), padding=(1, 1),output_padding=(0, 0), stride=(2, 2)),  # 4x4 -> 7x7
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_filters, hidden_filters, kernel_size=4, padding=0,output_padding=0, stride=2), # 7x7 -> 16x16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_filters, self.num_output_channels, kernel_size=2,padding=2,output_padding=0,stride=2), #16x16 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        if self.layer_type == 'mlp':
            x_recon = self.decoder(z)
            x_recon = x_recon.reshape(z.shape[0], self.out_dim[0], self.out_dim[1], self.out_dim[2])
        elif self.layer_type == 'cnn':
            x = self.linear(z)
            x = x.reshape([x.shape[0], -1, 7, 7])
            x_recon = self.convs_decoder(x)

        return x_recon
