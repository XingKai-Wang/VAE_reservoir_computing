import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from VAE import VAE
from helper import *
from datasets import datasets
import visdom
import torch.autograd



def training(name, model, optimizer, data_loader, epoch):
    print(model)
    training_loss = 0
    total_loss = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model.to(device)
    model.train()
    for e in range(epoch):
        epoch_loss = []
        for batch_index, data in enumerate(data_loader):
            train_data, _ = data
            # train_data: [b, 1, 28, 28]
            train_data = Variable(train_data)
            if torch.cuda.is_available():
                train_data = train_data.cuda()
            optimizer.zero_grad()
            loss, recon_img, reloss, kld = model(train_data)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            epoch_loss.append(loss.item())
            if batch_index % 140 == 139:
                print('[%d, %5d] loss: %.3f' %
                      (e + 1, batch_index + 1, training_loss / 140))
                training_loss = 0.0
        total_loss.append(np.mean(epoch_loss))

        if e % 10 == 0:
            visdom_visualization(name, model, train_loader)
    plot_loss(name, epoch, total_loss)

    img_grid = visualization_laten(model.decoder)
    save_image(img_grid, './plot/vae_laten.png')
    torch.save(model.state_dict(), './model/{}.pt'.format(name))

    return total_loss

if __name__ == '__main__':
    train_loader, val_loader, test_loader = datasets()
    model = VAE(activation = 'LeakyRelu')
    optimizer = optimizer(model, 1e-3)
    total_loss = training('first_mlp_training', model, optimizer, data_loader = train_loader, epoch = 20)


