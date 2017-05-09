import numpy as np

class SIGMOID_LAYER:
    """docstring forRELU_LAYER."""
    def __init__(self):
        pass

    def forward(self, X):
        """
        Computes the forward pass of Sigmoid Layer.
        Input:
            X: Input data of any shape
        """
        self.cache = X
        self.feature_map = 1.0/(1.0 + np.exp(-X))
        return self.feature_map, 0

    def backward(self, delta):
        """
        Computes the backward pass of Sigmoid Layer.
        Input:
            delta: Shape of delta values should be same as of X in cache
        """
        self.delta_X = delta * (self.feature_map) * (1 - self.feature_map)
        return self.delta_X
