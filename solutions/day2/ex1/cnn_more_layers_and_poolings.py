import torch

class CNN_More_Layers_And_Poolings(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # the convolutions
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(in_channels=24, out_channels=28, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(in_channels=28, out_channels=32, kernel_size=3, padding = 1)
        self.pool = torch.nn.MaxPool2d(2, 2)

        # the fully connected part of the network
        # after applying the convolutions and poolings, the tensor
        # has the shape 24 x 6 x 6, see below
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, self.n_classes)
        )
        self.activation = torch.nn.LogSoftmax(dim=1)

    def apply_convs(self, x):
      # input image has shape 3 x  32 x 32
      x = self.pool(torch.nn.functional.relu(self.conv1(x)))
      # shape after conv: 12 x 30 x 30
      # shape after pooling: 12 x 15 x 15
      x = self.pool(torch.nn.functional.relu(self.conv2(x)))
      # shape after conv: 24 x 13 x 13
      # shape after pooling: 24 x 6 x 6
      x = self.pool(torch.nn.functional.relu(self.conv3(x)))
      # shape after conv: 28 x 4 x 4
      # shape after pooling: 28 x 2 x 2
      x = self.pool(torch.nn.functional.relu(self.conv4(x)))
      # shape after conv: 32 x 2 x 2
      # shape after pooling: 32 x 1 x 1

      return x
    
    def forward(self, x):
        x = self.apply_convs(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        x = self.activation(x)
        return x

