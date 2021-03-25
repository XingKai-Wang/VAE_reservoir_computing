import torch
from torch.autograd import Variable
import numpy as np
from VAE import VAE
from helper import *
from datasets import datasets
#from datasets import Binarize
import visdom
import torch.autograd



def training(name, model, optimizer, train_loader, epoch):
    print(model)
    training_loss = 0
    total_loss = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model.to(device)
    model.train()
    for e in range(epoch + 1):
        epoch_loss = []
        for batch_index, data in enumerate(train_loader):
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

        viz = visdom.Visdom()
        x, _ = iter(train_loader).next()
        with torch.no_grad():
            _, x_hat, _, _ = model(x)
            viz.images(x, nrow=8, win='origin',opts=dict(title = 'x_origin'))
            viz.images(x_hat, nrow=8, win='recon', opts=dict(title='x_recon'))



    torch.save(model.state_dict(), './model/{}.pt'.format(name))

    return total_loss

if __name__ == '__main__':
    train_loader, val_loader, test_loader = datasets()
    model = VAE(activation = 'LeakyRelu')
    optimizer = optimizer(model, 1e-3)
    total_loss = training('first_mlp_training', model, optimizer, train_loader = train_loader, epoch = 1)


