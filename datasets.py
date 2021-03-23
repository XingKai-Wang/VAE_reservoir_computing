import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class Datasets:
    def __init__(self, mode = 'train'):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if mode == 'train':
            self.data = torchvision.datasets.MNIST(root = './data',
                                                   train = True,
                                                   download = True,
                                                   transform = self.transform)
        if mode == 'Test':
            self.data = torchvision.datasets.MNIST(root = './data',
                                                   train = False,
                                                   download = True,
                                                   transform = self.transform)

    def trainloader(self, data):
        mnist = DataLoader(data, batch_size = 16, shuffle = True, num_workers = 2)
        return mnist

    def testloader(self, data):
        mnist = DataLoader(data, batch_size = 16, shuffle = True, num_workers = 2)
        return mnist


