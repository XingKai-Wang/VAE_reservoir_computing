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
    def __init__(self, input_channels, hidden_channels, kernel_size, return_all_layers = False):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.kernel_size = kernel_size
        self.return_all_layers = return_all_layers
        all_layers = []

        for i in range(self.num_layers):
            # name = 'cell{}'.format(i)
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]
            cell = ConvLSTMCell(cur_input_channels, self.hidden_channels[i], self.kernel_size)
            # setattr(self, name, cell)
            all_layers.append(cell)

        self.cell_list = nn.ModuleList(all_layers)

    def forward(self, input, hidden_state = None):
        batch_size, _, _, h, w = input.size()
        if hidden_state is not None:
            raise NotImplementedError()
        else:   # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=batch_size, hidden = self.hidden_channels,image_size=(h,w))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input

        for i in range(self.num_layers):
            output_inner = []
            h, c = hidden_state[i]

            for seq_index in range(input.size(1)):
                h, c = self.cell_list[i](cur_layer_input[:, seq_index, :, :, :], h, c)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list


    def _init_hidden(self, batch_size, hidden, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size,  hidden[i], image_size))
        return init_states

if __name__ == '__main__':
    convlstm = ConvLSTM(input_channels=1, hidden_channels=[128, 64, 32], kernel_size=3).cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, 5, 1, 64, 64)).cuda()
    target = Variable(torch.randn(1, 5, 32, 64, 64)).double().cuda()

    output = convlstm(input).double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)

