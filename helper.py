import torch
import visdom
from scipy.stats import norm
from torchvision.utils import make_grid
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def visdom_visualization(name, model, data_loader):
    viz = visdom.Visdom(env = name)
    x, _ = iter(data_loader).next()
    with torch.no_grad():
        _, x_hat, _, _ = model(x)
        viz.images(x, nrow=8, win='origin', opts=dict(title='x_origin'))
        viz.images(x_hat, nrow=8, win='recon', opts=dict(title='x_recon'))

def plot_loss(name, epoch, total_loss):
    sns.set_style('darkgrid')
    x = np.arange(1, epoch + 1)
    my_x_ticks = np.arange(1, epoch + 1, 1)
    plt.xticks(my_x_ticks)
    sns.lineplot(x=x, y=total_loss, label='training loss on the mnist data')
    # plt.fill_between(x = 'nums', y1 = df['avg'] - df['std'], y2 = df['avg'] + df['std'], alpha = 0.1,color = 'blue', data = df)
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.legend()
    plt.savefig('./plot/{}_training_loss.png'.format(name))

def visualization_laten(decoder, grid_size = 20):
    start = 0.5 / (grid_size + 1)
    end = (grid_size + 0.5) / (grid_size + 1)

    x = torch.tensor(norm.ppf(torch.linspace(start, end, 20)))
    y = torch.tensor(norm.ppf(torch.linspace(start, end, 20)))

    laten = torch.stack(torch.meshgrid(x, y))
    laten = laten.reshape(-1, 10)

    img = decoder.forward(laten.float())
    #img = torch.sigmoid(mean)
    img_grid = make_grid(img, nrow=grid_size, padding=10)

    return img_grid
