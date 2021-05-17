import torch
import visdom
from scipy.stats import norm
from torchvision.utils import make_grid
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.utils import save_image

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

def evaluation(model, data_loader, total_loss_val,device):
    val_loss = 0.0
    epoch_loss = []
    model.eval()
    for batch_index, data in enumerate(data_loader):
        val_data, _ = data
        val_data = Variable(val_data)
        if torch.cuda.device_count() > 0:
            val_data = val_data.to(device)
        loss, recom_img, reloss, kld = model(val_data)

        val_loss += loss.item()
        epoch_loss.append(loss.item())
    total_loss_val.append(np.mean(epoch_loss))

    return total_loss_val

def evaluation_lstm(model, data_loader, total_loss_val,device):
    val_loss = 0.0
    epoch_loss = []
    model.eval()
    for batch_index, data in enumerate(data_loader):
        val_data = data
        val_data = Variable(val_data)
        if torch.cuda.device_count() > 0:
            val_data = val_data.to(device)
        loss, recom_img, reloss, kld = model(val_data)

        val_loss += loss.item()
        epoch_loss.append(loss.item())
    total_loss_val.append(np.mean(epoch_loss))

    return total_loss_val

def visdom_visualization(name, number, model, data_loader):
    viz = visdom.Visdom(env = name + number)
    x, _ = iter(data_loader).next()
    with torch.no_grad():
        _, x_hat, _, _ = model(x)
        viz.images(x, nrow=8, win='origin', opts=dict(title='x_origin'))
        viz.images(x_hat, nrow=8, win='recon', opts=dict(title='x_recon'))

def plot_movingmnist(recon_image):
    fig1 = plt.figure(1, figsize=(10, 5))
    for i in range(0, 20):
        # create plot
        toplot_pred = recon_image[0, i, :, :].squeeze(1).permute(1, 2, 0)
        plt.imshow(toplot_pred.cpu().detach().numpy())
        plt.savefig('../plot' + '/%i_image.png' % (i + 1))

def plot_loss(name, number, epoch, total_loss, eva_type):
    fig2 = plt.figure(2)
    sns.set_style('darkgrid')
    x = np.arange(0, epoch + 1)
    my_x_ticks = np.arange(0, epoch + 1, 5)
    plt.xticks(my_x_ticks)
    sns.lineplot(x=x, y=total_loss, label='{} on the mnist data'.format(eva_type))
    # plt.fill_between(x = 'nums', y1 = df['avg'] - df['std'], y2 = df['avg'] + df['std'], alpha = 0.1,color = 'blue', data = df)
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(eva_type))
    plt.legend()
    plt.savefig('../plot/{}{}_{}.png'.format(name, number, eva_type))

def visualization_laten(decoder, grid_size = 8):
    start = 0.5 / (grid_size + 1)
    end = (grid_size + 0.5) / (grid_size + 1)

    x = torch.tensor(norm.ppf(torch.linspace(start, end, 8)))
    y = torch.tensor(norm.ppf(torch.linspace(start, end, 8)))

    laten = torch.stack(torch.meshgrid(x, y))
    laten = laten.reshape(-1, 10)

    img = decoder.forward(laten.float())
    img_grid = make_grid(img, nrow=grid_size, padding=10)

    return img_grid
