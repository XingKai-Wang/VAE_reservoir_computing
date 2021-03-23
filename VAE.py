import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()
        self.Encoder = encoder_net

    @staticmethod
    def reparameterization(mu, log_var):
        '''
        :param mu:
        :param log_var:
        :return: reparameterization for Gussian
        '''
        std = torch.exp(0.5 * log_var)
        epsilon = torch.rand_like(std)

        return mu + epsilon * std

    def encode(self, x):
        '''
        divide outputs as mean and log variance
        :param x:
        :return: mean and log variance
        '''
        h_e = self.Encoder(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e




