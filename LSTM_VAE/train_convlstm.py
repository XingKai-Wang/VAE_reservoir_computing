import sys
sys.path.append('..')

import argparse
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from helper import *
from LSTM_VAE.MovingMNIST_dataset import *
from LSTM_VAE.ConvLSTM_VAE import *
import torch.autograd
from torchsummary import summary
import gc
from PytorchES import EarlyStopping




def training(args):
    # load datasets
    train_loader, val_loader, test_loader = processeddataset(path='../data/MovingMNIST/mnist_test_seq.npy', batch_size=args.batch_size)

    # create model
    model = LSTM_VAE(input_channels=args.input_channels, hidden_channels_e=args.hidden_channels_e, hidden_channels_d=args.hidden_channels_d, kernel_size=args.kernel_size,z_dim=args.z_dim)
    # optimizer
    optim = optimizer(model, args.lr)
    # schedular
    sche = schedular(optim)
    # intialize earlystoping
    early_stopping = EarlyStopping(patience=5, verbose=True)
    # start training
    training_loss = 0
    total_loss = []
    total_loss_val = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    summary(model, (20, 1, 64, 64))
    model.train()
    for e in range(args.epoch + 1):
        epoch_loss = []
        for batch_index, data in enumerate(train_loader):
            train_data = data
            # print(train_data.size())
            # train_data: [b, seq, 1, 64, 64]
            train_data = Variable(train_data)
            if torch.cuda.device_count() > 0:
                train_data = train_data.to(device)
            optim.zero_grad()
            loss, recon_img, reloss, kld = model(train_data)
            loss.backward()
            optim.step()

            training_loss += loss.item()
            epoch_loss.append(loss.item())
            if batch_index % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                      (e + 1, batch_index + 1, training_loss / 140))
                training_loss = 0.0
            del train_data
            gc.collect()
            #torch.cuda.empty_cache()
        total_loss.append(np.mean(epoch_loss))

    # start validation
        with torch.no_grad():
            total_loss_val, val_loss_es = evaluation_lstm(model,val_loader,total_loss_val,device)

        # sche.step()

        early_stopping(val_loss_es, model)

        if early_stopping.early_stop:
            with torch.no_grad():
                    plot_movingmnist(recon_img)
            print("Early stopping")
            break
    # plot reconstruct image
        with torch.no_grad():
            if e == args.epoch:
                plot_movingmnist(recon_img)

    plot_loss(args.model, args.number, total_loss, 'training loss')
    plot_loss(args.model, args.number, total_loss_val, 'validation loss')
    # plot laten space
    if args.z_dim == 2:
        img_grid = visualization_laten(model.decoder)
        save_image(img_grid, './plot/{}{}vae_laten.png'.format(args.model,args.number))
    torch.save(model.state_dict(), '../model/{}{}.pt'.format(args.model,args.number))

    return total_loss, total_loss_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# model hyperparameters
    parser.add_argument('--model',default='ConvLSTM',type=str,help='what model to use in VAE', choices=['ConvLSTM', 'RC'])
    parser.add_argument('--z_dim',default=20,type=int,help='dimension of laten space')
    parser.add_argument('--input_channels', default=128,type=int,help='input channels for decoder')
    parser.add_argument('--hidden_channels_e',default=[32, 64, 128], nargs='+',type=int,help='list contains hidden channels for different layer in encoder')
    parser.add_argument('--hidden_channels_d', default=[64, 32, 1], nargs='+',type=int,help='list contains hidden channels for different layer in decoder')
    parser.add_argument('--kernel_size',default=3,type=int,help='kernel size in ConvLSTM')

# optimizer hyperparameters
    parser.add_argument('--lr',default=1e-3,type=float,help='learning rate')
    parser.add_argument('--batch_size',default=64,type=int,help='batch size')

# other hyperparameters
    parser.add_argument('--epoch',default=10,type=int,help='number of epochs')
    parser.add_argument('--number',default=0,type=str,help='index of different model')

    args = parser.parse_args()
    training(args)
