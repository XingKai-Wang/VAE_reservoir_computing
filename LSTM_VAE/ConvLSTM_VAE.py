import torch
import torch.nn as nn
from LSTM_VAE.ConvLSTM_Encoder_Decoder import ConvLSTMEncoder, ConvLSTMDecoder
from LSTM_VAE.Sequence_Reservoir_Encoder_Decoder import ReserviorEncoder, ReservoirDecoder
from helper import *
import numpy as np

class LSTM_VAE(nn.Module):
    def __init__(self, model, input_channels, hidden_channels_e, hidden_channels_d, kernel_size, num_filters, z_dim):
        super(LSTM_VAE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels_e = hidden_channels_e
        self.hidden_channels_d = hidden_channels_d
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.z_dim = z_dim

        if model == 'ConvLSTM':
            self.encoder = ConvLSTMEncoder(self.hidden_channels_e, self.kernel_size, self.z_dim)
            self.decoder = ConvLSTMDecoder(self.input_channels, self.hidden_channels_d, self.kernel_size, self.z_dim // 2)
        if model == 'RC':
            self.encoder = ReserviorEncoder(input_dim=64*64, reservoir_dim=1024, num_filters=self.num_filters, z_dim=self.z_dim)
            self.decoder = ReservoirDecoder(z_dim=self.z_dim // 2, num_filters=self.num_filters, output_channels=1)

    def forward(self, image):
        # encode forward process: calculate the mu and log_var of z
        mu, log_var = self.encoder.forward(image)
        # calculate KL divergence
        # kld_total = 0
        # latent_z = []
        # for mu_log in mu_logvar:
        #     mu, log_var = mu_log
        #     kld_total += KLD(mu, log_var) / (image.size(0) * 64 * 64)
        #     # apply reparameterization trick on each z
        #     z = reparameterization(mu, log_var)
        #     latent_z.append(z)

        kld = KLD(mu, log_var) / (image.size(0) * 64 * 64)
        z = reparameterization(mu, log_var)
        # decode forward process: reconstruct data with 64 x 64 dimensions
        recon_img = self.decoder.forward(z)
        # sample a binary image
        image = torch.bernoulli(image)
        # calculate the reconstruction loss
        criterion = nn.BCELoss()
        # print(recon_img.shape)
        # print(torch.max(recon_img), torch.min(recon_img))
        # print(torch.max(image), torch.min(image))
        reloss = criterion(recon_img, image)
        # elbo  = kld + reloss
        elbo = kld + reloss

        return elbo, recon_img, reloss, kld
