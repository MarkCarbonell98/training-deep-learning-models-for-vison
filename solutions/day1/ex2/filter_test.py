import os
import numpy as np
import sys
sys.path.append(os.path.abspath('utils'))
import matplotlib.pyplot as plt
import torch
import utils
from functools import partial
from logistic_regressor_conv_filters import gaussian_kernel

laplacian_filter = torch.Tensor([[0,1,0], [1,-4,1], [0,1,0]])
gaussian_filter = gaussian_kernel(dim=3)

imgs, labels = utils.load_cifar(os.path.join('./cifar10', 'test'))

trafos = [utils.to_channel_first, utils.normalize, utils.to_tensor]

trafos = partial(utils.compose, transforms=trafos)

data = utils.DatasetWithTransform(imgs, labels, transform=trafos)
loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)

conv_gaussian = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, bias=False)
conv_gaussian.weight.data[:,:] = gaussian_filter

conv_laplacian = torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, bias=False)
conv_laplacian.weight.data[:,:] = laplacian_filter

conv_laplacian_alone = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, bias=False)
conv_laplacian_alone.weight.data[:,:] = laplacian_filter

conv_gaussian2 = torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, bias=False)
conv_gaussian2.weight.data[:,:] = gaussian_filter


fig = plt.figure(figsize=(15, 15))
with torch.no_grad():
    for batch, (x,y) in enumerate(loader):
        pred_gaussian = conv_gaussian(x).cpu()[0,0:3].permute(1,2,0)
        ax1 = fig.add_subplot(151)
        ax1.imshow(pred_gaussian)
        ax1.set_title("Prediction Gaussian")

        ax3 = fig.add_subplot(152)
        pred_laplacian = conv_laplacian_alone(x).cpu()[0,0:3].permute(1,2,0)
        ax3.set_title("Pred Laplacian Alone")
        ax3.imshow(pred_laplacian)

        ax4 = fig.add_subplot(153)
        ax4.set_title("Gaussian + Laplacian")
        pred_both = conv_laplacian(conv_gaussian(x)).cpu()[0, 0:3].permute(1,2,0)
        ax4.imshow(pred_both)

        ax5 = fig.add_subplot(154)
        ax5.set_title("Laplacian + Gaussian")
        pred_both_inv = conv_gaussian2(conv_laplacian_alone(x)).cpu()[0, 0:3].permute(1,2,0)
        ax5.imshow(pred_both_inv)

        ax2 = fig.add_subplot(155)
        ax2.imshow(x[0,...].permute(1,2,0))
        ax2.set_title("Original")

plt.show()
