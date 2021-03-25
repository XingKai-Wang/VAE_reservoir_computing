import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np


def datasets(root = './data', download = True, batch_size = 64, num_workers = 4):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5,])])

    datasets = torchvision.datasets.MNIST(root = root,
                                      train = True,
                                      download = download,
                                      transform = transform)
    test_data = torchvision.datasets.MNIST(root = root,
                                          train = False,
                                          download = download,
                                          transform = transform)

    train_data = data.dataset.Subset(datasets, np.arange(45000))
    val_data = data.dataset.Subset(datasets, np.arange(45000, 50000))


    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    return train_loader, val_loader, test_loader


