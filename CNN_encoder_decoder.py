import torch
import torch.nn as nn

class CNNEncoder:
    def __init__(self, num_input_channels = 1, num_filters = 32, z_dims = 20):
        '''

        :param num_input_channels: Number of input channels of the image. For MNIST, this parameter is 1
        :param num_filters: number of filters to use in first layer
        :param z_dims: dimensions of laten space
        '''
        super(CNNEncoder, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.z_dim = z_dims

        hidden_filters = num_filters
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.num_filters, kernel_size=(4,4), padding=(1,1), stride=(2,2)), # 28x28 -> 14x14
            nn.LeakyReLU(),
            nn.Conv2d(hidden_filters, 2 * hidden_filters, kernel_size=(4,4),padding=(1,1),stride=(2,2)), # 14x14 -> 7x7
            nn.LeakyReLU(),
            nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size=(3,3),padding=(1,1),stride=(1,1)), # 7x7 -> 7x7
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4 * num_filters * 7 * 7, z_dims)
        )

    def forward(self,x):
        h_e = self.encoder(x)
        mu, log_var = torch.chunk(h_e, 2, dim=1)
        return mu, log_var

class CNNDecoder:
    def __init__(self, num_input_channels = 1, num_filters = 32, z_dims = 10):
        super(CNNDecoder, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.z_dim = z_dims

        hidden_filters = num_filters
        self.linear = nn.Sequential(
            nn.Linear(z_dims, 4 * hidden_filters * 7 * 7),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            # output=(input - 1) * stride + output_padding - 2 * padding + kernel
            nn.ConvTranspose2d(4 * hidden_filters, 2 * hidden_filters, kernel_size=(3,3), padding=(1,1), output_padding=(0,0), stride=(1,1)), # 7x7 -> 7x7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2 * hidden_filters, hidden_filters, kernel_size=(4,4), padding=(1,1), output_padding=(0,0), stride=(2,2)), # 7x7 -> 14x14
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_filters, self.num_input_channels, kernel_size=(4,4), padding=(1,1), output_padding=(0,0), stride=(2,2)), # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.reshape([x.shape[0], -1, 7, 7])
        x_recon = self.decoder(x)
        return x_recon