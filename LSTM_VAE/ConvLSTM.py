import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from LSTM_VAE.MovingMNIST_dataset import *


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        # input cell
        self.wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # forget cell
        self.wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # memory cell
        self.wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # output cell
        self.wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.wci = None
        self.wcf = None
        self.wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.wxi(x) + self.whi(h) + self.wci * c)
        cf = torch.sigmoid(self.wxf(x) + self.whf(h) + self.wcf * c)
        cc = cf * c + ci * torch.tanh(self.wxc(x) + self.whc(h))
        co = torch.sigmoid(self.wxo(x) + self.who(h) + self.wco * cc)
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.wci is None:
            self.wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.wci.size()[2], 'Input width mismatched'
            assert shape[1] == self.wci.size()[3], 'Input height mismatched'

        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.kernel_size = kernel_size
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for seq_index in range(input.size(1)):
            x = input[:,seq_index,:,:]
            for i in range(self.num_layers):
                # intialize all cells in first step
                name = 'cell{}'.format(i)
                if seq_index == 0:
                    batch_size, _, h, w = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size, self.hidden_channels[i], (h, w))

                    internal_state.append((h, c))
                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            outputs.append(x)

        return outputs, (x, new_c)

if __name__ == '__main__':
    train_loader, val_loader, test_loader = processeddataset('../data/MovingMNIST/mnist_test_seq.npy')
    convlstm = ConvLSTM(input_channels=1, hidden_channels=[128, 64, 32], kernel_size=3).cuda()

    for batch_index, data in enumerate(train_loader):
        train_data = data
        output = convlstm(train_data.cuda())
        print(torch.stack(output[0], 1).shape)
        break

