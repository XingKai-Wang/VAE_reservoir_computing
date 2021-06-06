import torch
import torch.nn as nn
import echotorch.nn as etnn
import echotorch.utils.matrix_generation as mg
from torch.autograd import Variable


from LSTM_VAE.MovingMNIST_dataset import *

if __name__ == '__main__':
    w_generator = mg.NormalMatrixGenerator()
    wbias_generator = mg.NormalMatrixGenerator()
    win_generator = mg.NormalMatrixGenerator()

    esn = etnn.LiESN(input_dim=4096, hidden_dim=2048, output_dim=4096, leaky_rate=0.9, w_generator=w_generator,
                     win_generator=win_generator, wbias_generator=wbias_generator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        esn.to(device)

    train_loader, val_loader, test_loader = processeddataset('../data/MovingMNIST/mnist_test_seq.npy',batch_size=16)
    for batch_index, data in enumerate(train_loader):
        train_data = data
        train_data = train_data.reshape(train_data.size(0), train_data.size(1), train_data.size()[3] * train_data.size()[4])
        train_data = Variable(train_data)
        if torch.cuda.device_count() > 0:
            train_data.to(device)

        esn(train_data)

