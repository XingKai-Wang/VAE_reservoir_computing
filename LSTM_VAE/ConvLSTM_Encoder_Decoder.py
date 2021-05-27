import torch
import torch.nn as nn
import numpy as np
from LSTM_VAE.ConvLSTM import ConvLSTM


class ConvLSTMEncoder(nn.Module):
    def __init__(self, hidden_channels, kernel_size, z_dim):
        super(ConvLSTMEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.z_dim = z_dim

        self.convlstm = ConvLSTM(1, self.hidden_channels, self.kernel_size)
        self.linear = nn.Linear(self.hidden_channels[-1] * 64 * 64, self.z_dim)

    def forward(self, x):
        outputs = self.convlstm(x)
        # for output in outputs:
        #         #     h_e = self.linear(output.view(-1, self.hidden_channels[-1] * 64 * 64))
        #         #     (mu, log_var) = torch.chunk(h_e, 2, dim=1)
        #         #     mu_logvar.append((mu, log_var))
        h_e = self.linear(outputs.view(outputs.size(0), 20, -1, self.hidden_channels[-1] * 64 * 64))
        mu, log_var = torch.chunk(h_e, 2, dim=3)

        return mu, log_var

class ConvLSTMDecoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, z_dim):
        super(ConvLSTMDecoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.z_dim = z_dim

        self.linear = nn.Linear(self.z_dim, self.input_channels * 64 * 64)
        self.convlstm = ConvLSTM(self.input_channels, self.hidden_channels,self.kernel_size)
        self.conv3d = nn.Conv3d(self.hidden_channels[-1], 1, kernel_size=3, stride=1,padding=1)

    def forward(self, z):
        x = self.linear(z.view(z.size(0), 20, -1, self.z_dim))
        x = x.reshape(x.size(0), 20, -1, 64, 64)
        # B C S H W
        x = self.convlstm(x).permute(0, 2, 1, 3, 4)
        # B S C H W
        x_recon = torch.sigmoid(self.conv3d(x)).permute(0, 2, 1, 3, 4)

        return x_recon
