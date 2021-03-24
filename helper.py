import torch


def reparameterization(mu, log_var):
    '''
    :param mu:
    :param log_var:
    :return: reparameterization for Gussian
    '''
    std = torch.exp(0.5 * log_var)
    epsilon = torch.rand_like(std)

    return mu + epsilon * std

def KLD(mu, log_var):
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    kld = torch.sum(kld_element).mul_(-0.5)

    return kld
