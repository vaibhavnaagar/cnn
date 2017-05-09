import matplotlib.pyplot as plt
import numpy as np

def plots(x, y, z, steps):
    try:
        plt.figure(1)
        plt.plot(x, '-bo', label="Loss")
        plt.xlabel('Number of iterations', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.title('Training Error rate vs Number of iterations')
        plt.savefig("Loss_function_vs_iter_batch_128.jpeg")
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
        plt.savefig("error_rates_batch_128.jpeg")
    except:
        pass
    pass


arr_files1 = np.load("128/loss_history.npz")
arr_files2 = np.load("128/step_loss_history.npz")

plots(arr_files1['arr_0'], arr_files2['arr_0'], arr_files2['arr_1'], np.arange(len(arr_files2['arr_1'])))
