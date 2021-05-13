import torch
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset, random_split, SubsetRandomSampler, DataLoader


def MovingMNISTdataloader(path):
    '''
    load MovingMNIST dataset, shape = [time sequence, batch size, width, height]
    B S H W -> S B H W
    :param path: path of the MovingMnist dataset
    :return: transpose dataset
    '''
    data = np.load(path)
    data_trans = data.transpose(1, 0, 2, 3)
    return data_trans

class MovingMNISTdataset(Dataset):
    '''
    output shape: B S C=1 H W
    '''
    def __init__(self, path):
        self.data = MovingMNISTdataloader(path)


    def __len__(self):
        return len(self.data[:, 0, 0, 0])

    def __getitem__(self, index):
        self.trainsample_ = self.data[index, ...]
        self.sample = self.trainsample_ / 255.0 # Normalization
        self.sample = torch.from_numpy(np.expand_dims(self.trainsample_, axis = 1)).float()

        return self.sample

def processeddataset(path, valid_size = 0.2, batch_size = 16, num_workers = 4, shuffer = True):
    # load movingmnist dataset
    movingmnist = MovingMNISTdataset(path)
    # split train_dataset and test_dataset by 0.8 : 0.2
    train_size = int(len(movingmnist) * 0.8)
    test_size = len(movingmnist) - train_size
    torch.manual_seed(torch.initial_seed())
    train_dataset, test_dataset = random_split(movingmnist, [train_size, test_size])
    # split training_set and validation_set
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_index, val_index = indices[split:], indices[:split]
    # define sampler, no need to pre-shuffle because the indices will be shuffled during each pass
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler,num_workers=num_workers)
    val_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=val_sampler,num_workers=num_workers)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffer,num_workers=num_workers)

    return train_loader, val_loader, test_loader

