import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dims = 784, hidden_dims = (512, 256), z_dims = 20, activation = 'Relu'):
        '''
        :param input_dims: input dimensions of mnist: 784
        :param hidden_dims: two linear layers with 512 and 256 dimensions
        :param z_dims: dimensions of z: 20
        '''
        super(MLPEncoder, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.z_dims = z_dims
        self.activation = activation
        self.hidden_layer_Relu = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.z_dims)
        )
        self.hidden_layer_Leakyrelu = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dims[1], self.z_dims)
        )

    def encode(self, x):
        h_e = None
        if self.activation == 'Relu':
            h_e = self.hidden_layer_Relu(x.view(-1, 784))
        elif self.activation == 'LeakyRelu':
            h_e = self.hidden_layer_Leakyrelu(x.view(-1, 784))

        return h_e

    def forward(self, x):
        '''
        divide outputs as mean and log variance
        :param x: input images
        :return: mean and log variance
        '''
        h_e = self.encode(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e

class MLPDecoder(nn.Module):
    def __init__(self, z_dims = 10, hidden_dims = (256, 512), output_dims = (1, 28, 28), activation = 'Relu'):
        '''

        :param z_dims: dimensions of z: 20
        :param hidden_dims: two linear layer with 256, 512 dimensions
        :param output_dims: reconstruct data with 784 dimensions (28 * 28)
        '''
        super(MLPDecoder, self).__init__()
        self.z_dims = z_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.activation = activation
        self.hidden_layer_Relu = nn.Sequential(
            nn.Linear(self.z_dims, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.output_dims[1] * output_dims[2]),
            nn.Sigmoid()
        )
        self.hidden_layer_Leakyrelu = nn.Sequential(
            nn.Linear(self.z_dims, self.hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dims[1], self.output_dims[1] * self.output_dims[2]),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_recon = None
        if self.activation == 'Relu':
            x_recon = self.hidden_layer_Relu(z)
        elif self.activation == 'LeakyRelu':
            x_recon = self.hidden_layer_Leakyrelu(z)
        x_recon = x_recon.reshape(z.shape[0], self.output_dims[0], self.output_dims[1], self.output_dims[2])

        return x_recon






