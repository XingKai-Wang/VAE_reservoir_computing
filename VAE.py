import torch
import torch.nn as nn
from helper import *
from MLP_encoder_decoder import MLPEncoder
from MLP_encoder_decoder import MLPDecoder
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = MLPEncoder()
        self.decoder = MLPDecoder()

    def forward(self, image):
        # encode forward process: calculate the mu and log_var of z
        mu, log_var = self.encoder.forward(image)
        # calculate KL divergence
        kld = torch.sum(KLD(mu, log_var))
        # apply reparameterization trick on z
        z = reparameterization(mu, log_var)
        # decode forward process: reconstruct data with 784 dimensions
        recon_img = self.decoder.forward(z)
        # calculate the reconstruction loss
        reloss = F.binary_cross_entropy_with_logits(recon_img, image, reduction = 'sum')
        # elbo  = kld + reloss
        elbo = (kld + reloss).mean()

        return elbo, recon_img


