import torch.nn as nn

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
        x = x.view(-1, self.n_pixels)
        x = self.log_reg(x)
        return x
