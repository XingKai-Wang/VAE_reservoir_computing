from LSTM_VAE.MovingMNIST_dataset import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_loader, val_loader, test_loader = processeddataset('../data/MovingMNIST/mnist_test_seq.npy')
    for batch_index, data in enumerate(train_loader):
        if batch_index == 1:
            break
        train_data = data
        print(train_data.shape)
        print(torch.max(train_data))
        print(torch.min(train_data))

        # for i in range(0, 20):
        #     # create plot
        #     fig = plt.figure(figsize=(10, 5))
        #     toplot_pred = train_data[0,i,:,:].squeeze(1).permute(1,2,0)
        #     plt.imshow(toplot_pred)
        #     plt.savefig('../plot' + '/%i_image.png' % (i + 1))
