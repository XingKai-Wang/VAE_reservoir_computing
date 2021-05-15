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
        mu_logvar = []
        outputs = self.convlstm(x)[0]
        for output in outputs:
            h_e = self.linear(output.view(-1, self.hidden_channels[-1] * 64 * 64))
            (mu, log_var) = torch.chunk(h_e, 2, dim=1)
            mu_logvar.append((mu, log_var))

        return mu_logvar

class ConvLSTMDecoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, z_dim):
        super(ConvLSTMDecoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.z_dim = z_dim

        self.linear = nn.Linear(self.z_dim, self.input_channels * 64 * 64)
        self.convlstm = ConvLSTM(self.input_channels, self.hidden_channels,self.kernel_size)

    def forward(self, z):
        for i in range(len(z)):
            z[i] = self.linear(z[i]).reshape(z[i].shape[0], -1, 64, 64)
        z = torch.stack(z, dim=1)
        x_recon = self.convlstm(z)

        return x_recon[0]
