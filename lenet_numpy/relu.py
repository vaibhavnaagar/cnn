import numpy as np

class RELU_LAYER:
    """docstring forRELU_LAYER."""
    def __init__(self):
        pass

    def forward(self, X):
        """
        Computes the forward pass of Relu Layer.
        Input:
            X: Input data of any shape
        """
        self.cache = X
        self.feature_map = np.maximum(X, 0)
        return self.feature_map, 0

    def backward(self, delta):
        """
        Computes the backward pass of Relu Layer.
        Input:
            delta: Shape of delta values should be same as of X in cache
        """
        self.delta_X = delta * (self.cache >= 0)
        return self.delta_X
