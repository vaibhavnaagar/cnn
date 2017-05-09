from conv import *
from relu import *
from sigmoid import *
from fc import *
from maxpool import *
from softmax import *
from tsne_img_plot import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import timeit
from itertools import chain
from scipy import misc

class LENET5:
    """docstring forLENET5."""
    def __init__(self, t_input, t_output, v_input, v_output):
        """
        Creates Lenet-5 architecture
        Input:
            t_input: True Training input of shape (N, Depth, Height, Width)
            t_output: True Training output of shape (N, Class_Number)
        """
        b_dir = "16/"
        # Conv Layer-1
        conv1 = CONV_LAYER((6, 28, 28), (6, 1, 5, 5), (784, 4704), pad=2, stride=1, filename=b_dir+"conv0.npz")
        relu1 = RELU_LAYER()
        # Sub-sampling-1
        pool2 = MAX_POOL_LAYER(stride=2)
        # Conv Layer-2
        conv3 = CONV_LAYER((16, 10, 10), (16, 6, 5, 5), (1176, 1600), pad=0, stride=1, filename=b_dir+"conv3.npz")
        relu3 = RELU_LAYER()
        # Sub-sampling-2
        pool4 = MAX_POOL_LAYER(stride=2)
        # Fully Connected-1
        fc5 = FC_LAYER(120, (400, 120), filename=b_dir+"fc6.npz")
        sigmoid5 = SIGMOID_LAYER()
        # Fully Connected-2
        fc6 = FC_LAYER(84, (120, 84), filename=b_dir+"fc8.npz")
        sigmoid6 = SIGMOID_LAYER()
        # Fully Connected-3
        output = FC_LAYER(10, (84, 10), filename=b_dir+"fc10.npz")
        softmax = SOFTMAX_LAYER()
        self.layers = [conv1, relu1, pool2, conv3, relu3, pool4, fc5, sigmoid5, fc6, sigmoid6, output, softmax]


        """
        #Check gradient on smaller network. (Time is precious :)

        fc1 = FC_LAYER(5, (3, 5))
        sigmoid1 = SIGMOID_LAYER()
        fc2 = FC_LAYER(3, (5, 3))
        sigmoid2 = SOFTMAX_LAYER()

        temp = np.sin(np.arange(1,21)).reshape(5,4, order='F')/10
        fc1.kernel = temp[:, 1:4].T
        fc1.bias = temp[:, 0]
        print("FC1:::::::::::::::::")
        print(fc1.kernel)
        print(fc1.bias)

        temp = np.sin(np.arange(1,19)).reshape(3,6, order='F')/10
        fc2.kernel = temp[:, 1:6].T
        fc2.bias = temp[:, 0]
        print("FC2:::::::::::::::::")
        print(fc2.kernel)
        print(fc2.bias)
        self.layers = [fc1, sigmoid1, fc2, sigmoid2]
        """

        """
        # Only FC layers
        fc1 = FC_LAYER(120, (784, 120))
        sigmoid1 = RELU_LAYER()
        # Fully Connected-2
        fc2 = FC_LAYER(60, (120, 60))
        sigmoid2 = RELU_LAYER()
        # Fully Connected-3
        output = FC_LAYER(10, (60, 10))
        softmax = SOFTMAX_LAYER()

        self.layers = [fc1, sigmoid1, fc2, sigmoid2, output, softmax]
        """

        self.X = t_input
        self.Y = t_output
        self.Xv = v_input
        self.Yv = v_output

    @staticmethod
    def one_image_time(X, layers):
        """
        Computes time of conv and fc layers
        Input:
            X: Input
            layers: List of layers.
        Output:
            inp: Final output
        """
        inp = X
        conv_time = 0.0
        fc_time = 0.0
        layer_time = []

        for layer in layers:
            start = timeit.default_timer()
            if isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)
            stop = timeit.default_timer()
            layer_time += [stop-start]
            if isinstance(layer, (FC_LAYER, SIGMOID_LAYER, SOFTMAX_LAYER)):
                fc_time += stop - start
            if isinstance(layer, (CONV_LAYER, RELU_LAYER)):
                conv_time += stop - start
        return conv_time, fc_time, layer_time

    @staticmethod
    def visualize_feature_maps(X, layers, digit, batch_string):
        """
        Create and save image of feature maps of conv and fc layers
        Input:
            X: Input an image of shape (1, 1, 28, 28)
            layers: List of layers.
        """
        inp = X
        size = (224,224)
        misc.imsave("feature_maps/" + digit + "/" + "input_" + digit + batch_string + ".jpeg", misc.imresize(X[0][0], size))
        conv_i = 1
        max_i = 1

        for layer in layers:
            if isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)

            if isinstance(layer, RELU_LAYER):
                for channel in range(inp.shape[1]):
                    misc.imsave("feature_maps/" + digit + "/" + "conv" + str(conv_i) + "_c" + str(channel+1)+ batch_string + ".jpeg", misc.imresize(inp[0][channel], size))
                conv_i += 1

            if isinstance(layer, MAX_POOL_LAYER):
                for channel in range(inp.shape[1]):
                    misc.imsave("feature_maps/" + digit + "/" + "maxpool" + str(max_i) + "_c" + str(channel+1)+ batch_string + ".jpeg", misc.imresize(inp[0][channel], size))
                max_i += 1

    @staticmethod
    def feedForward(X, layers):
        """
        Computes final output of neural network by passing
        output of one layer to another.
        Input:
            X: Input
            layers: List of layers.
        Output:
            inp: Final output
        """
        inp = X
        wsum = 0
        for layer in layers:
            if isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)
            wsum += ws
        return inp, wsum

    @staticmethod
    def backpropagation(Y, layers):
        """
        Computes final output of neural network by passing
        output of one layer to another.
        Input:
            Y: True output
            layers: List of layers.
        Output:
            grad: gradient
        """
        delta = Y
        for layer in layers[::-1]:
            delta = layer.backward(delta)

    @staticmethod
    def update_parameters(layers, batch_size, a, z, m):
        """
        Update weight parameters of each layer
        """
        for layer in layers:
            if isinstance(layer, (CONV_LAYER, FC_LAYER)):
                layer.update_kernel(batch=batch_size, alpha=a, zeta=z, method=m)

    @staticmethod
    def loss_function(pred, t, **params):
        """
        Computes loss using cross-entropy method.
        Input:
            pred: Predicted output of network of shape (N, C)
            t: true output of shape (N, C)
            w_sum: sum of squares of all weight parameters for L2 regularization
        where,
            N: batch size
            C: Number of classes in the final layer
        Output:
            Loss or cost
        """
        w_sum = params.get("wsum", 0)
        #print("w_sum: ", w_sum)
        z = params.get("zeta", 0)
        assert t.shape == pred.shape
        #print("Shape: ", t.shape, z)
        epsilon = 1e-10
        return ((-t * np.log(pred + epsilon)).sum() + (z/2)*w_sum) / pred.shape[0]

    @staticmethod
    def plots(x, y, z, steps):
        try:
            plt.figure(1)
            plt.plot(x, '-bo', label="Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss', fontsize=18)
            plt.title('Training Error rate vs Number of iterations')
            plt.savefig("Loss_function_vs_iter.jpeg")
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
            plt.savefig("error_rates.jpeg")
        except:
            pass
        pass

    def lenet_train(self, **params):
        """
        Train the Lenet-5.
        Input:
            params: parameters including "batch", "alpha"(learning rate),
                    "zeta"(regularization parameter), "method" (gradient method),
                    "epochs", ...
        """
        batch  = params.get("batch", 50)             # Default 50
        alpha  = params.get("alpha", 0.01)            # Default 0.1
        zeta   = params.get("zeta", 0)               # Default 0 (No regularization)
        method = params.get("method", "adam")            # Default
        epochs = params.get("epochs", 4)             # Default 4
        print("Training on params: batch=", batch, " learning rate=", alpha, " L2 regularization=", zeta, " method=", method, " epochs=", epochs)
        self.loss_history = []
        self.gradient_history = []
        self.valid_loss_history = []
        self.step_loss = []
        print(method)
        X_train = self.X
        Y_train = self.Y
        assert X_train.shape[0] == Y_train.shape[0]
        num_batches = int(np.ceil(X_train.shape[0] / batch))
        step = 0;
        steps = []
        X_batches = zip(np.array_split(X_train, num_batches, axis=0), np.array_split(Y_train, num_batches, axis=0))

        for ep in range(epochs):
            print("Epoch: ", ep, "===============================================")
            for x, y in X_batches:
                predictions, weight_sum = LENET5.feedForward(x, self.layers)
                loss = LENET5.loss_function(predictions, y, wsum=weight_sum, zeta=zeta)
                self.loss_history += [loss]
                LENET5.backpropagation(y, self.layers)          #check this gradient
                LENET5.update_parameters(self.layers, x.shape[0], alpha, zeta, method)
                print("Step: ", step, ":: Loss: ", loss, "weight_sum: ", weight_sum)
                if step % 100 == 0:
                    pred, w = LENET5.feedForward(self.Xv, self.layers)
                    v_loss = LENET5.loss_function(pred, self.Yv, wsum=w, zeta=zeta)
                    print("Validation error: ", v_loss)
                    steps += [step]
                    self.valid_loss_history += [v_loss]
                    self.step_loss += [loss]
                step += 1

            XY = list(zip(X_train, Y_train))
            np.random.shuffle(XY)
            new_X, new_Y = zip(*XY)
            assert len(new_X) == X_train.shape[0] and len(new_Y) == len(new_X)
            X_batches = zip(np.array_split(new_X, num_batches, axis=0), np.array_split(new_Y, num_batches, axis=0))
        np.savez("step_loss_history", self.step_loss, self.valid_loss_history)
        np.savez("loss_history", self.loss_history)
        LENET5.plots(self.loss_history, self.step_loss, self.valid_loss_history, steps)
        pass

    def lenet_predictions(self, X, Y):
        """
        Predicts the ouput and computes the accuracy on the dataset provided.
        Input:
            X: Input of shape (Num, depth, height, width)
            Y: True output of shape (Num, Classes)
        """
        start = timeit.default_timer()
        predictions, weight_sum = LENET5.feedForward(X, self.layers)
        stop = timeit.default_timer()

        loss = LENET5.loss_function(predictions, Y, wsum=weight_sum, zeta=0.99)
        y_true = np.argmax(Y, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        print("Dataset accuracy: ", accuracy_score(y_true, y_pred)*100)
        print("FeedForward time:", stop - start)
        pass

    def save_parameters(self):
        """
        Saves the weights and biases of Conv and Fc layers in a file.
        """
        for layer in self.layers:
            if isinstance(layer, CONV_LAYER):
                np.savez("conv" + str(self.layers.index(layer)), layer.kernel, layer.bias)
            elif isinstance(layer, FC_LAYER):
                np.savez("fc" + str(self.layers.index(layer)), layer.kernel, layer.bias)
        pass

    def check_gradient(self):
        """
        Computes the numerical gradient and compares with Analytical gradient
        """
        sample = 10
        epsilon = 1e-4
        X_sample = self.X[range(sample)]
        Y_sample = self.Y[range(sample)]
        predictions, weight_sum = LENET5.feedForward(X_sample, self.layers)
        LENET5.backpropagation(Y_sample, self.layers)

        abs_diff = 0
        abs_sum = 0

        for layer in self.layers:
            if not isinstance(layer, (CONV_LAYER, FC_LAYER)):
                continue
            i = 0
            print("\n\n\n\n\n")
            print(type(layer))
            del_k = layer.delta_K + (0.99*layer.kernel/sample)
            kb = chain(np.nditer(layer.kernel, op_flags=['readwrite']), np.nditer(layer.bias, op_flags=['readwrite']))
            del_kb = chain(np.nditer(del_k, op_flags=['readonly']), np.nditer(layer.delta_b, op_flags=['readonly']))

            for w, dw in zip(kb, del_kb):
                w += epsilon
                pred, w_sum = LENET5.feedForward(X_sample, self.layers)
                loss_plus = LENET5.loss_function(pred, Y_sample, wsum=w_sum, zeta=0.99)

                w -= 2*epsilon
                pred, w_sum = LENET5.feedForward(X_sample, self.layers)
                loss_minus = LENET5.loss_function(pred, Y_sample, wsum=w_sum, zeta=0.99)

                w += epsilon
                numerical_gradient = (loss_plus - loss_minus)/(2*epsilon)

                abs_diff += np.square(numerical_gradient - dw)
                abs_sum  += np.square(numerical_gradient + dw)
                print(i, "Numerical Gradient: ", numerical_gradient, "Analytical Gradient: ", dw)
                if not np.isclose(numerical_gradient, dw, atol=1e-4):
                    print("Not so close")
                if i >= 10:
                    break
                i += 1

        print("Relative difference: ", np.sqrt(abs_diff)/np.sqrt(abs_sum))
        pass

    def tsne_plot(self, X, y, name):
        activations, dummy = LENET5.feedForward(X, self.layers[0:10])
        print(activations.shape)
        if activations.shape != (10000, 84):
            import sys
            sys.exit()
        plot_tsne(activations, y, name)
