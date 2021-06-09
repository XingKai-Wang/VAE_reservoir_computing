import torch.nn as nn
from torchesn.nn import ESN
from torch.autograd import Variable
from LSTM_VAE.MovingMNIST_dataset import *


def reshape_input(input_data):
    input_data = input_data.squeeze().reshape(input_data.size(0), input_data.size(1), input_data.size()[3] * input_data.size()[4])
    return input_data.transpose(0,1)

input_size = 4096
hidden_size = 2048
output_size = 4096
washout_rate = 0.1

if __name__ == '__main__':

    model = ESN(input_size, hidden_size, output_size, output_steps='mean', readout_training='gd',batch_first=False,nonlinearity='relu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)

    train_loader, val_loader, test_loader = processeddataset('../data/MovingMNIST/mnist_test_seq.npy',batch_size=8)
    for batch_index, data in enumerate(train_loader):
        train_data = data
        train_data = reshape_input(train_data)
        washout_list = [0] * train_data.size(1)
        # train_data = Variable(train_data)
        if torch.cuda.device_count() > 0:
            train_data = train_data.to(device)

        output, hidden = model(train_data, washout_list)
        break






