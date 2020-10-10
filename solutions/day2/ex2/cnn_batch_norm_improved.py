import torch

class CNNBatchNormImproved(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # the convolutions
        self.conv1 =torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5)
        self.conv2 =torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        # the pooling layer
        self.pool =torch.nn.MaxPool2d(2, 2)
        # the normalization layers
        self.bn1 =torch.nn.BatchNorm2d(12)
        self.bn2 =torch.nn.BatchNorm1d(24)
        self.instance_norm1 = torch.nn.InstanceNorm2d(12)
        self.instance_norm2 = torch.nn.InstanceNorm2d(24)

        self.normalize = torch.nn.Sequential(
                self.instance_norm1,
                self.instance_norm2
        )

        # the fully connected part of the network
        # after applying the convolutions and poolings, the tensor
        # has the shape 24 x 6 x 6, see below
        self.fc =torch.nn.Sequential(
           torch.nn.Linear(24 * 6 * 6, 120),
           torch.nn.ReLU(),
           torch.nn.Linear(120, 60),
           torch.nn.ReLU(),
           torch.nn.Linear(60, self.n_classes)
        )
        self.activation =torch.nn.LogSoftmax(dim=1)

    def apply_convs(self, x):
        # input image has shape 3 x  32 x 32
        x = self.pool(torch.nn.functional.relu(self.normalize(self.conv1(x))))
        # shape after conv: 12 x 28 x 28                                   
        # shape after pooling: 12 x 14 X 14                                
        x = self.pool(torch.nn.functional.relu(self.normalize(self.conv2(x))))
        # shape after conv: 24 x 12 x 12
        # shape after pooling: 24 x 6 x 6
        return x
    
    def forward(self, x):
        x = self.apply_convs(x)
        x = x.view(-1, 24 * 6 * 6)
        x = self.fc(x)
        x = self.activation(x)
        return x

