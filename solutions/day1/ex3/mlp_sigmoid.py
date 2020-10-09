import torch

class MLP_Sigmoid(torch.nn.Module):
    def __init__(self, n_pixels, n_classes):
        super().__init__()
        self.n_pixels = n_pixels
        self.n_classes = n_classes
        
        # here, we define the structure of the MLP.
        # it's imporant that we use a non-linearity after each 
        # fully connected layer! Here we use the rectified linear
        # unit, short ReLu
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_pixels, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_classes),
            torch.nn.LogSigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, self.n_pixels)
        x = self.layers(x)
        return x

