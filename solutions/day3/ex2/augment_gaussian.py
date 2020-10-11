import torch


class AugmentGaussian:
    def __init__(self, train_stddev_rng_range):
        assert len(train_stddev_rng_range) == 2
        self.minval, self.maxval = train_stddev_rng_range
        self.minval = self.minval / 255
        self.maxval = self.maxval / 255

    def __call__(self, x):
        rng_stddev = (self.maxval - self.minval) * torch.rand(1) + self.minval
        return x + torch.randn(x.size()) * rng_stddev

