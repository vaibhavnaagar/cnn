from mnist import MNIST
import os, sys
import numpy as np
from lenet5 import *
import timeit

class LoadMNISTdata:
    """docstring for LoadMNISTdata."""
    lim = 256.0

    def __init__(self, data_path):
        self.path = data_path

    def loadData(self):
        mndata = MNIST(self.path)
        train_img, train_label = mndata.load_training()
        test_img, test_label = mndata.load_testing()
        self.train_img = np.asarray(train_img, dtype='float64') / LoadMNISTdata.lim
        self.train_label = np.asarray(train_label)
        self.test_img = np.asarray(test_img, dtype='float64') / LoadMNISTdata.lim
        self.test_label = np.asarray(test_label)

        print("train_img:", self.train_img.shape)
        print("train_label:", self.train_label.shape)
        print("test_img:", self.test_img.shape)
        print("test_label:", self.test_label.shape)


if __name__ == '__main__':
    cwd = os.getcwd()
    dataset = LoadMNISTdata(cwd)
    dataset.loadData()
    N = 50000
    #small_data = dataset.train_img[range(100)]
    #X_train = small_data.reshape(100, 1, 28,28)
    X_train = dataset.train_img[range(0, N)].reshape(N, 1, 28, 28)
    Y_train = np.zeros((N, 10))
    Y_train[np.arange(N), dataset.train_label[range(0, N)]] = 1

    M = 10000
    X_valid = dataset.train_img[N:].reshape(M, 1, 28, 28)
    Y_valid = np.zeros((M, 10))
    Y_valid[np.arange(M), dataset.train_label[N:]] = 1
    print("Validation set: ", X_valid.shape, Y_valid.shape)
    print("Training set: ", X_train.shape, Y_train.shape)

    ### Create LeNet5 object ###
    mylenet = LENET5(X_train, Y_train, X_valid, Y_valid)

    ### Check Gradients ###
    #mylenet.check_gradient()

    ### GET conv and fc layers time ###
    #print(LENET5.one_image_time(X_train[0].reshape(1, 1, 28, 28), mylenet.layers))

    ### Training LENET5 ###
    """
    start = timeit.default_timer()
    mylenet.lenet_train(method="adam", epochs=4, batch=64, alpha=0.001, zeta=0)
    stop = timeit.default_timer()

    print("Training time:", stop - start)
    ### Save kernel and bias of conv and fc layers ###
    mylenet.save_parameters()

    ### Training Set accuracy ###
    print("Training ", end="")
    mylenet.lenet_predictions(X_train, Y_train)
    """

    ### Test set accuracy ###
    X_test = dataset.test_img.reshape(10000, 1, 28, 28)
    Y_test = np.zeros((10000, 10))
    Y_test[np.arange(10000), dataset.test_label] = 1
    print("Test ", end="")
    mylenet.lenet_predictions(X_test, Y_test)

    batch_string = "_batch_16"
    ### Visualize Feature Maps of conv layers for a image of a digit ###
    #for digit, index in zip(range(10),[3, 2, 1, 18, 4, 8, 11, 0, 61, 7]):
    #    mylenet.visualize_feature_maps(X_test[index].reshape(1, 1, 28, 28), mylenet.layers, str(digit), batch_string)

    ### Merge each channel image of a conv layers ###
    #for digit in range(10):
    #    merge_images(digit, batch_string)


    ### Plot TSNE of all Test dataset images ###
    print("Plotting TSNE for", batch_string)
    mylenet.tsne_plot(X_test, dataset.test_label, "combined_feature_maps/tsne_plots/tsne_batch_16.jpeg")


    """
    Check gradients on smaller network.
    X = np.sin(np.arange(1,16)).reshape(5,3, order='F')/10
    print("X:::::::::::::::::")
    print(X)

    Y = np.zeros((5, 3))
    Y[np.arange(5), [1,2,0,1,2]] = 1
    print(Y)

    ml = LENET5(X, Y)
    ml.check_gradient()
    """
