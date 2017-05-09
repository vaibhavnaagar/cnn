from  tsne_img_plot import *
import numpy as np
from mnist import MNIST


def plots(x, y, z, steps):
        try:
            plt.figure(1)
            plt.plot(x, '-bo', label="Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss', fontsize=18)
            plt.title('Training Error rate vs Number of iterations')
            plt.savefig("Loss_function_vs_iter_b_16.jpeg")
        except:
            pass

        try:
            plt.figure(2)
            plt.plot(steps, y, '-bo', label="Training Loss")
            plt.plot(steps, z, '-ro', label="Validation Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss Value', fontsize=18)
            plt.title('Training and Validation error rates vs number of iterations')
            plt.legend(loc='upper right')
            plt.savefig("error_rates_b_16.jpeg")
        except:
            pass
        pass


def main():
	cost = np.loadtxt("cost_b_16.mat")
	validation = np.loadtxt("valid_b_16.mat")
	fmaps = np.loadtxt("featuremap_b_16.mat")
	print(cost.shape, validation.shape)
	mndata = MNIST("../")
	test_img, test_label = mndata.load_testing()
	test_label = np.asarray(test_label)
	plots(cost, validation[:, 1], validation[:,2],validation[:, 0])
	print("Plotting, tsne...")
	plot_tsne(fmaps, test_label, "mlp_tsne_plot_b_16.jpeg")

if __name__ == '__main__':
	main()
