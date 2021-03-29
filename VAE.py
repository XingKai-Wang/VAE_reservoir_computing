import torch
import torch.nn as nn
from helper import *
from MLP_encoder_decoder import MLPEncoder
from MLP_encoder_decoder import MLPDecoder
import torch.nn.functional as F
from torch import optim

class VAE(nn.Module):
    def __init__(self, activation = None):
        super(VAE, self).__init__()
        self.activation = activation
        self.encoder = MLPEncoder(input_dims = 784, hidden_dims = (512, 256), z_dims = 20, activation = self.activation)
        self.decoder = MLPDecoder(z_dims = 10, hidden_dims = (256, 512), output_dims = (1, 28, 28), activation = self.activation)

    def forward(self, image):
        # encode forward process: calculate the mu and log_var of z
        mu, log_var = self.encoder.forward(image)
        # calculate KL divergence
        kld = KLD(mu, log_var) / (image.size(0) * 28 * 28)
        # apply reparameterization trick on z
        z = reparameterization(mu, log_var)
        # decode forward process: reconstruct data with 784 dimensions
        recon_img = self.decoder.forward(z)
        # sample a binary image
        image = torch.bernoulli(image)
        # calculate the reconstruction loss
        criterion = nn.BCELoss()
        reloss = criterion(recon_img, image)
        # elbo  = kld + reloss
        elbo = kld + reloss

        return elbo, recon_img, reloss, kld


