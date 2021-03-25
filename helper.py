import torch
from scipy.stats import norm
from torchvision.utils import make_grid
from torch import optim

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

def optimizer(model, lr):
    optimizer = optim.Adam(model.parameters(), lr = lr)
    return optimizer

def visualiztion_laten(decoder, gird_size = 20):
    start = 0.5 / (gird_size + 1)
    end = 1 / (gird_size + 1)

    x = torch.tensor(norm.ppf(torch.linspace(start, end, 20)))
    y = torch.tensor(norm.ppf(torch.linspace(start, end, 20)))

    laten = torch.stack(torch.meshgrid(x, y))
    laten = laten.reshape(-1, 2)

    mean = decoder.forward(laten)
    img = torch.sigmoid(mean)
    img_grid = make_grid(img)

    return img_grid
