import numpy as np

class SOFTMAX_LAYER:
    """docstring forRELU_LAYER."""
    def __init__(self):
        pass

    def forward(self, X):
        """
        Computes the forward pass of Softmax Layer.
        Input:
            X: Input data of shape (N, C)
        where,
            N: Batch size
            C: Number of nodes in SOFTMAX_LAYER or classes
        Output:
            Y: Final output of shape (N, C)
        """
        self.cache = X
        dummy = np.exp(X)
        self.Y = dummy/np.sum(dummy, axis=1, keepdims=True)
        return self.Y, 0

    def backward(self, output):
        """
        Computes the backward pass of Softmax Layer.
        Input:
            output: Training set ouput of shape (N, C)
        """
        #assert self.Y.shape == output.shape
        self.delta_X =  (self.Y - output) / self.Y.shape[0]
        return self.delta_X

    def softmax_loss(self, Y, output):
        """
        Computes loss using cross-entropy method.
        Input:
            Y: Predicted output of network of shape (N, C)
            output: real output of shape (N, C)
        where,
            N: batch size
            C: Number of classes in the final layer
        """
        assert Y.shape == output.shape
        epsilon = 1e-10
        self.loss = (-output * np.log(Y + epsilon)).sum() / Y.shape[0]
        pass
