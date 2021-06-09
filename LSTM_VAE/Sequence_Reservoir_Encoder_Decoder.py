import torch
import torch.nn as nn
from LSTM_VAE.ConvLSTM import ConvLSTM
from torchesn.nn import ESN

class ReserviorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, z_dim, output_step='mean', nonlinearity='relu'):
        super(ReserviorEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.output_setp = output_step
        self.nonlinearity = nonlinearity

        self.esn = ESN(self.input_dim,self.hidden_dim,self.output_dim,output_steps=self.output_setp,nonlinearity=self.nonlinearity)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.output_dim, self.z_dim)
            )


    def forward(self, x):
        x = x.squeeze().reshape(x.size(0),x.size(1),x.size()[3] *x.size()[4]).transpose(0,1)
        washout_list = [0] * x.size(1)
        h, _ = self.esn(x, washout_list)
        h_e = self.linear(h)
        mu, log_var = torch.chunk(h_e, 2, dim=2)

        return mu, log_var

class ReservoirDecoder(nn.Module):
    def  __init__(self, z_dim, output_dim, hidden_channels, kernel_size):
        super(ReservoirDecoder, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size


        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, self.output_dim),
            nn.ReLU()
        )

        self.convlstm = ConvLSTM(1, self.hidden_channels, self.kernel_size)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(self.hidden_channels[-1], 1, kernel_size=3, stride=1,padding=1)
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.reshape(x.size(0), x.size(1), 1, 64, 64).transpose(0,1)
        x = self.convlstm(x).permute(0, 2, 1, 3, 4)
        x_recon = torch.sigmoid(self.decoder(x)).permute(0, 2, 1, 3, 4)

        return x_recon

