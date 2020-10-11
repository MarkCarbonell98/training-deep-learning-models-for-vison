
import torch
from skimage.util import random_noise


class AugmentPoisson:
    def __init__(self, train_stddev_rng_range):
        assert len(train_stddev_rng_range) == 2
        self.minval, self.maxval = train_stddev_rng_range
        self.minval = self.minval / 255
        self.maxval = self.maxval / 255

    def __call__(self, x):
        img = random_noise(x, mode="poisson", seed=1)
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        return img
