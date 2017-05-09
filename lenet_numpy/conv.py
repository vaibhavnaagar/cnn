import numpy as np
import sys
import scipy.signal

class CONV_LAYER:
    """docstring forCONV_LAYER."""
    def __init__(self, layer_size, kernel_size, fan, **params):
        """
        Input:
            layer_size: tuple consisting (depth, height, width)
            kernel_size: tuple consisting (number_of_kernels, inp_depth, inp_height, inp_width)
            fan: tuple of number of nodes in previous layer and this layer
            params: directory consists of pad_len and stride,
                    filename (to load weights from file)
        """
        self.depth, self.height, self.width = layer_size

        fname = params.get("filename", None)
        if fname:
            try:
                arr_files = np.load(fname)
                self.kernel = arr_files['arr_0']
                self.bias = arr_files['arr_1']
                assert np.all(self.kernel.shape == kernel_size) and np.all(self.bias.shape[0] == kernel_size[0])
            except:
                raise
        else:
            f = np.sqrt(6)/np.sqrt(fan[0] + fan[1])
            epsilon = 1e-6
            self.kernel = np.random.uniform(-f, f + epsilon, kernel_size)
            self.bias = np.random.uniform(-f, f + epsilon, kernel_size[0])

        self.pad = params.get('pad', 0)                  # Default 0
        self.stride = params.get('stride', 1)            # Default 1
        if self.pad < 0:
            print("Invalid padding: pad cannot be negative")
            sys.exit()
        self.gradient_history = np.zeros(kernel_size)
        self.bias_history = np.zeros(kernel_size[0])
        self.m_kernel = np.zeros(kernel_size)
        self.m_bias = np.zeros(kernel_size[0])
        self.v_kernel = np.zeros(kernel_size)
        self.v_bias = np.zeros(kernel_size[0])
        self.timestamp = 0
        pass

    def forward(self, X):
        """
        Computes the forward pass of Conv Layer.
        Input:
            X: Input data of shape (N, D, H, W)
        Variables:
            kernel: Weights of shape (K, K_D, K_H, K_W)
            bias: Bias of each filter. (K)
        where, N = batch_size or number of images
               H, W = Height and Width of input layer
               D = Depth of input layer
               K = Number of filters/kernels or depth of this conv layer
               K_H, K_W = kernel height and Width

        Output:
        """
        pad_len = self.pad
        stride = self.stride

        N, D, H, W = X.shape
        K, K_D, K_H, K_W = self.kernel.shape

        #assert self.depth == K
        #assert D == K_D
        #assert K == self.bias.shape[0]

        conv_h = (H - K_H + 2*pad_len) // stride + 1
        conv_w = (W - K_W + 2*pad_len) // stride + 1

        #assert self.height == conv_h
        #assert self.width == conv_w
        #assert (H - K_H + 2*pad_len)%stride == 0
        #assert (W - K_W + 2*pad_len)%stride == 0

        # feature map of a batch
        self.feature_map = np.zeros([N, K, conv_h, conv_w])

        X_padded = np.pad(X, ((0,0), (0,0), (pad_len, pad_len), (pad_len, pad_len)), 'constant')

        if stride == 1:
            # scipy.signal.convolve2d doesn't have any attribute for stride, so it works for only stride = 1
            # Rotate kernel by 180
            kernel_180 = np.rot90(self.kernel, 2, (2,3))
            for img in range(N):
                for conv_depth in range(K):
                    for inp_depth in range(D):
                        self.feature_map[img, conv_depth] += scipy.signal.convolve2d(X_padded[img, inp_depth], kernel_180[conv_depth, inp_depth], mode='valid')
                    self.feature_map[img, conv_depth] += self.bias[conv_depth]
        else:
            # Manual convolution when stride > 1, but above method is faster.
            for img in range(N):
                for conv_depth in range(K):
                    for h in range(0, H + 2*pad_len - K_H + 1, stride):
                        for w in range(0, W + 2*pad_len - K_W + 1, stride):
                            self.feature_map[img, conv_depth, h//stride, w//stride] = \
                                np.sum(X_padded[img, :, h:h+K_H, w:w+K_W] * self.kernel[conv_depth,:,:,:]) + self.bias[conv_depth]

        self.cache = X
        return self.feature_map, np.sum(np.square(self.kernel))


    def backward(self, delta):
        """
        Computes the backward pass of Conv layer.
        Input:
            delta: derivatives from next layer of shape (N, K, conv_h, conv_w)
        """

        X = self.cache
        pad_len = self.pad
        stride = self.stride

        N, D, H, W = X.shape
        K, K_D, K_H, K_W = self.kernel.shape

        #assert self.depth == K
        #assert D == K_D
        #assert K == self.bias.shape[0]

        conv_h = (H - K_H + 2*pad_len) // stride + 1
        conv_w = (W - K_W + 2*pad_len) // stride + 1

        #assert self.height == conv_h
        #assert self.width == conv_w

        # Rotate Kernel by 180 degrees      [No need]
        #kernel_180 = np.rot90(kernel, 2, (2,3))
        X_padded = np.pad(X, ((0,0), (0,0), (pad_len, pad_len), (pad_len, pad_len)), 'constant')
        delta_X_padded = np.zeros(X_padded.shape)
        self.delta_K = np.zeros(self.kernel.shape)
        self.delta_b = np.zeros(self.bias.shape)

        # Delta X
        for img in range(N):
            for conv_depth in range(K):
                for h in range(0, H + 2*pad_len - K_H + 1, stride):
                    for w in range(0, W + 2*pad_len - K_W + 1, stride):
                        delta_X_padded[img, :, h:h+K_H, w:w+K_W] += delta[img, conv_depth, h//stride, w//stride] * self.kernel[conv_depth]

        if pad_len > 0:
            self.delta_X = delta_X_padded[:, :, pad_len:-pad_len, pad_len:-pad_len]
        else:
            self.delta_X = delta_X_padded[:]

        #assert self.delta_X.shape == X.shape

        # Delta kernel
        for img in range(N):
            for kernel_num in range(K):
                for h in range(conv_h):
                    for w in range(conv_w):
                        self.delta_K[kernel_num,:,:,:] += delta[img, kernel_num, h, w] * X_padded[img, :, h*stride:h*stride+K_H, w*stride:w*stride+K_W]

        # Delta Bias
        self.delta_b = np.sum(delta, (0,2,3))
        return self.delta_X

    def update_kernel(self, **params):
        """
        Update weights and biases stored in this layer.
        Input:
            params: Optional parameters- method, alpha, zeta
        """
        method = params.get("method", "")
        alpha = params.get("alpha", 0.001)
        zeta   = params.get("zeta", 0.01)
        batch_size = params.get("batch", 1)
        beta1        = params.get("beta1", 0.9)
        beta2        = params.get("beta2", 0.999)
        fudge_factor = 1e-8                         # smoothing term to avoid division by zero

        if method == "adagrad":
            self.gradient_history += np.square(self.delta_K + (zeta*self.kernel/batch_size))
            self.bias_history += np.square(self.delta_b)
            self.kernel -= alpha*(self.delta_K + (zeta*self.kernel/batch_size))/(np.sqrt(self.gradient_history) + fudge_factor)
            self.bias    -= alpha*self.delta_b/(np.sqrt(self.bias_history) + fudge_factor)
        elif method == "adam":
            self.timestamp += 1
            alpha = alpha * np.sqrt(1 - np.power(beta2, self.timestamp)) / (1 - np.power(beta1, self.timestamp))
            self.m_kernel = beta1 * self.m_kernel + (1 - beta1) * (self.delta_K + (zeta*self.kernel/batch_size))
            self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.delta_b
            self.v_kernel = beta2 * self.v_kernel + (1 - beta2) * np.square((self.delta_K + (zeta*self.kernel/batch_size)))
            self.v_bias = beta2 * self.v_bias + (1 - beta2) * np.square(self.delta_b)

            self.kernel -= np.divide(alpha * self.m_kernel, (np.sqrt(self.v_kernel) + fudge_factor))
            self.bias -= np.divide(alpha * self.m_bias, (np.sqrt(self.v_bias) + fudge_factor))
        else:
            self.kernel -= alpha*(self.delta_K + zeta*self.kernel/batch_size)
            self.bias   -= alpha*self.delta_b
        pass
