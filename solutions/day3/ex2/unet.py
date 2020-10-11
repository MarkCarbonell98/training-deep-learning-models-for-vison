import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
    """
    
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with leaky ReLU activation
    def _conv_block(self, in_channels, out_channels, block_num):
        conv_blocks = []
        for i in range(block_num):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels
            # add convolutional layer
            conv_blocks.append(nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1))
            # add batchnorm for better training stability
            conv_blocks.append(nn.BatchNorm2d(out_channels))
            # add activation function
            conv_blocks.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            
        return nn.Sequential(*conv_blocks)       


    # upsampling via nearest-neighbor interpolation
    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')
    
    # we do use a final Sigmoid activation this time, since we're dealing with a regression problem
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 5
        self.depth = 5
        
        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList
        
        # modules of the encoder path
        self.encoder = nn.ModuleList([self._conv_block(in_channels, 48, 2),
                                      self._conv_block(48, 48, 1),
                                      self._conv_block(48, 48, 1),
                                      self._conv_block(48, 48, 1),
                                      self._conv_block(48, 48, 1)])
        # the base convolution block
        self.base = self._conv_block(48, 48, 1)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(96, 96, 2),
                                      self._conv_block(144, 96, 2),
                                      self._conv_block(144, 96, 2),
                                      self._conv_block(144, 96, 2),
                                      self._conv_block(144, 64, 2)])
        
        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.depth)])
        
        # output conv with linear activation
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)
        
        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            # get the spatial dimension of the corresponding encoder features
            size = encoder_out[level].size()[2:]
            x = self._upsample(x, size)
            x = self.decoder[level](torch.cat((x, encoder_out[level]), dim=1))
        
        # apply output conv
        x = self.out_conv(x)
        return x

