import argparse
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from VAE import VAE
from helper import *
from datasets import datasets
import visdom
import torch.autograd




def training(args):
    # load datasets
    train_loader, val_loader, test_loader = datasets(batch_size=args.batch_size)

    # create model
    model = VAE(model_name=args.model,z_dim=args.z_dim,num_filters=args.num_filters,activation=args.activation,T=args.T,layer_type=args.layer_type)

    # optimizer
    optim = optimizer(model, args.lr)

    # start training
    print(model)
    training_loss = 0
    total_loss = []
    total_loss_val = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    model.train()
    for e in range(args.epoch + 1):
        epoch_loss = []
        for batch_index, data in enumerate(train_loader):
            train_data, _ = data
            # train_data: [b, 1, 28, 28]
            train_data = Variable(train_data)
            if torch.cuda.device_count() > 0:
                train_data = train_data.to(device)
            optim.zero_grad()
            loss, recon_img, reloss, kld = model(train_data)
            loss.backward()
            optim.step()

            training_loss += loss.item()
            epoch_loss.append(loss.item())
            if batch_index % 140 == 139:
                print('[%d, %5d] loss: %.3f' %
                      (e + 1, batch_index + 1, training_loss / 140))
                training_loss = 0.0
        total_loss.append(np.mean(epoch_loss))

    # start validation
        total_loss_val = evaluation(model,val_loader,total_loss_val,device)


    # plot loss and reconstruct image
        if e % 10 == 0:
            #visdom_visualization(args.model, args.number, model, train_loader)
            model.sample_and_save(args.batch_size, e, device)
    plot_loss(args.model, args.number, args.epoch, total_loss, 'training loss')
    plot_loss(args.model, args.number, args.epoch, total_loss_val, 'validation loss')
    # plot laten space
    if args.z_dim == 2:
        img_grid = visualization_laten(model.decoder)
        save_image(img_grid, './plot/{}{}vae_laten.png'.format(args.model,args.number))
    torch.save(model.state_dict(), './model/{}{}.pt'.format(args.model,args.number))

    return total_loss, total_loss_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# model hyperparameters
    parser.add_argument('--model',default='MLP',type=str,help='what model to use in VAE', choices=['MLP','CNN','RC','Casual'])
    parser.add_argument('--layer_type',default='cnn',type=str,help='what layer to use in RC encoder-decoder',choices=['mlp','cnn'])
    parser.add_argument('--z_dim',default=20,type=int,help='dimension of laten space')
    parser.add_argument('--num_filters',default=32,type=int,help='number of filters')
    parser.add_argument('--activation',default='LeakyRelu',type=str,help='what activate function to use',choices=['Relu','LeakyRelu'])
    parser.add_argument('--T',default=10,type=int,help='recurrent time')


# optimizer hyperparameters
    parser.add_argument('--lr',default=1e-3,type=float,help='learning rate')
    parser.add_argument('--batch_size',default=64,type=int,help='batch size')

# other hyperparameters
    parser.add_argument('--epoch',default=10,type=int,help='number of epochs')
    parser.add_argument('--number',default=0,type=str,help='index of different model')

    args = parser.parse_args()
    training(args)

