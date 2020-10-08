import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt

def gaussian_kernel(dim=3, mean = .5, std=.5):
    x = torch.arange(dim, dtype=torch.float32)
    x_grid = x.repeat(dim).view(dim, dim)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    var = std ** 2
    res = (1. / (2. * math.pi * var)) * torch.exp( - torch.sum(xy_grid - mean, dim=-1) ** 2 / (2 * var))
    res /= torch.sum(res)
    return res

# define logistic regression model
class LogisticRegressor(nn.Module):
    def __init__(self, n_pixels, n_classes):
        # the parent class 'nn.Module' needs to be initialised so
        # that all members that are subclasses of nn.Module themselves
        #  are correctly handled in autodiff
        super().__init__()
        # self.n_pixels = n_pixels
        self.n_pixels = n_pixels
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, bias=False, padding=1)

        with torch.no_grad():
            gaussian_filter = gaussian_kernel(dim=3)
            laplacian_filter = torch.Tensor([[0,1,0], [1,-4,1], [0,1,0]])
            self.conv1.weight.data[:,:] = gaussian_filter
            self.conv2.weight.data[:,:] = laplacian_filter


        self.filters = nn.Sequential(
                self.conv1,
                self.conv2
        )

        # nn.Sequential applies its arguments one after the other 
        self.log_reg = nn.Sequential(
            # nn.Linear instantiates a fully connected layer, the first argument
            # specifies the number of input units, the second argument the number
            # of output units
            nn.Linear(self.n_pixels, self.n_classes),
            # logarithmic softmax activation.
            # the combination of LogSoftmax and negative log-likelihood loss
            # (see below) corresponds to training with Cross Entropy, but is
            # numerically more stable
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        # reshape the input to be 1d instead of 2d,
        # which is required for fully connected layers
        x = self.filters(x)
        x = x.view(-1, self.n_pixels)
        x = self.log_reg(x)
        return x
