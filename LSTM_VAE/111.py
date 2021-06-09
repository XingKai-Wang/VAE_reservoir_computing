from tqdm import tqdm
from LSTM_VAE.MovingMNIST_dataset import *
import matplotlib.pyplot as plt
import imageio
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from helper import *
from LSTM_VAE.ConvLSTM_VAE import *

if __name__ == '__main__':
    train_loader, val_loader, test_loader = processeddataset('../data/MovingMNIST/mnist_test_seq.npy', batch_size=1)
    model = LSTM_VAE(model='ConvLSTM',hidden_channels_d=[128],hidden_channels_e=[128],input_channels=128,kernel_size=3,num_filters=32,z_dim=20)
    # checkpoint = torch.load('../model/ConvLSTM3.pt')
    # model.load_state_dict(checkpoint['model'])
    model.load_state_dict(torch.load('../model/checkpoint_128_8x8.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    model.eval()
    for batch_index, data in enumerate(train_loader):
        if batch_index == 1:
            break
        test_data = data
        # train_data = Variable(train_data)
        if torch.cuda.device_count() > 0:
            test_data = test_data.to(device)
        _, recon_image, _, _ = model(test_data)

        # create plot
        # fig = plt.figure(figsize=(10, 5))
        # for i in range(20):
        plot_movingmnist(test_data, 'test')
        plot_movingmnist(recon_image, 'recon')




