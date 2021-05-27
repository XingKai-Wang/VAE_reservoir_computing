from LSTM_VAE.MovingMNIST_dataset import *
import matplotlib.pyplot as plt
import imageio
import torch

if __name__ == '__main__':
    # train_loader, val_loader, test_loader = processeddataset('../data/MovingMNIST/mnist_test_seq.npy')
    # for batch_index, data in enumerate(train_loader):
    #     if batch_index == 1:
    #         break
    #     train_data = data
    #     print(train_data.shape)
    #     print(torch.max(train_data))
    #     print(torch.min(train_data))
    #     train_data = torch.bernoulli(train_data)
    #     for i in range(0, 20):
    #         # create plot
    #         fig = plt.figure(figsize=(10, 5))
    #         toplot_pred = train_data[0,i,:,:].squeeze()
    #         # print(toplot_pred.shape)
    #         plt.imshow(toplot_pred)
    #         plt.savefig('../plot' + '/%i_image.png' % (i + 1))


    im1 = imageio.imread('../plot/moving_mnist_plot/20_image.png')
    image1 = torch.bernoulli(torch.from_numpy(np.array(im1) / 255.0)).numpy()
    im2 = imageio.imread('../plot/RCS0/20_image.png')
    image2 = torch.bernoulli(torch.from_numpy(np.array(im2) / 255.0)).numpy()
    plt.imshow(image2)
    plt.show()
