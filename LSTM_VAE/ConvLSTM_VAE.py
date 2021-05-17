import torch
import torch.nn as nn
from LSTM_VAE.ConvLSTM_Encoder_Decoder import ConvLSTMEncoder, ConvLSTMDecoder
from helper import *
import numpy as np

class LSTM_VAE(nn.Module):
    def __init__(self, input_channels, hidden_channels_e, hidden_channels_d, kernel_size, z_dim):
        super(LSTM_VAE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels_e = hidden_channels_e
        self.hidden_channels_d = hidden_channels_d
        self.kernel_size = kernel_size
        self.z_dim = z_dim

        self.encoder = ConvLSTMEncoder(self.hidden_channels_e, self.kernel_size, self.z_dim)
        self.decoder = ConvLSTMDecoder(self.input_channels, self.hidden_channels_d, self.kernel_size, self.z_dim // 2)

    def forward(self, image):
        # encode forward process: calculate the mu and log_var of z
        mu_logvar = self.encoder.forward(image)
        # calculate KL divergence
        kld_total = 0
        latent_z = []
        for mu_log in mu_logvar:
            mu, log_var = mu_log
            kld_total += KLD(mu, log_var) / (image.size(0) * 64 * 64)
            # apply reparameterization trick on each z
            z = reparameterization(mu, log_var)
            latent_z.append(z)

        kld = kld_total / 20.0
        # decode forward process: reconstruct data with 64 x 64 dimensions
        recon_img = self.decoder.forward(latent_z)
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
