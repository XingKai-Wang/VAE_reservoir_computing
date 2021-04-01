import torch
import torch.nn as nn
from helper import *
from MLP_encoder_decoder import MLPEncoder
from MLP_encoder_decoder import MLPDecoder
from CNN_encoder_decoder import CNNEncoder
from CNN_encoder_decoder import CNNDecoder
from torchvision.utils import save_image

class VAE(nn.Module):
    def __init__(self, model_name, z_dim, num_filters, activation, *args, **kwargs):
        super(VAE, self).__init__()
        self.activation = activation
        self.z_dim = z_dim
        if model_name == 'MLP':
            self.encoder = MLPEncoder(input_dims=784, hidden_dims=(512, 256), z_dims=z_dim, activation=activation)
            self.decoder = MLPDecoder(z_dims=z_dim // 2, hidden_dims=(256, 512), output_dims=(1, 28, 28), activation=activation)
        if model_name == 'CNN':
            self.encoder = CNNEncoder(num_input_channels=1, num_filters=num_filters, z_dims=z_dim)
            self.decoder = CNNDecoder(num_input_channels=1, num_filters=num_filters,z_dims=z_dim // 2)

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

    def sample_and_save(self, batch_size, epoch):
        rand = torch.randn(batch_size, self.z_dim // 2)
        x = self.decoder.forward(rand)
        x_samples = (torch.rand(x.shape) < x) * 1

        img_grid = make_grid(x_samples, nrow=8).float()
        save_image(img_grid, "./plot/grid_{}.png".format(epoch))



