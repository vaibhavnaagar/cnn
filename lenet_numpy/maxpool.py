import numpy as np

class MAX_POOL_LAYER:
    """MAX_POOL_LAYER only reduce dimensions of height and width by a factor.
       It does not put max filter on same input twice i.e. stride = factor = kernel_dimension
    """
    def __init__(self, **params):
        self.factor = params.get('stride', 2)

    def forward(self, X):
        """
        Computes the forward pass of MaxPool Layer.
        Input:
            X: Input data of shape (N, D, H, W)
        where, N = batch_size or number of images
               H, W = Height and Width of input layer
               D = Depth of input layer
        """
        factor = self.factor
        N, D, H, W = X.shape
        #assert H%factor == 0 and W%factor == 0
        self.cache = [X, factor]
        self.feature_map = X.reshape(N, D, H//factor, factor, W//factor, factor).max(axis=(3,5))
        #assert self.feature_map.shape == (N, D, H//factor, W//factor)
        return self.feature_map, 0

    def backward(self, delta):
        """
        Computes the backward pass of MaxPool Layer.
        Input:
            delta: delta values of shape (N, D, H/factor, W/factor)
        """
        X, factor = self.cache
        if len(delta.shape) != 4:           # then it must be 2
            #assert delta.shape[0] == X.shape[0]
            delta = delta.reshape(self.feature_map.shape)

        fmap = np.repeat(np.repeat(self.feature_map, factor, axis=2), factor, axis=3)
        dmap = np.repeat(np.repeat(delta, factor, axis=2), factor, axis=3)
        #assert fmap.shape == X.shape and dmap.shape == X.shape

        self.delta_X = np.zeros(X.shape)
        self.delta_X = (fmap == X) * dmap

        #assert self.delta_X.shape == X.shape
        return self.delta_X
