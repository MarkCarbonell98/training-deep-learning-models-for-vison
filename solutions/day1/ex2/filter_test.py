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
gaussian_filter = gaussian_kernel(dim=3, channels=3)

imgs, labels = utils.load_cifar(os.path.join('./cifar10', 'test'))

plt.imshow(imgs[0])

trafos = [utils.to_channel_first, utils.normalize, utils.to_tensor]

trafos = partial(utils.compose, transforms=trafos)

data = utils.DatasetWithTransform(imgs, labels, transform=trafos)
loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)

conv = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, bias=False, padding=1)
print(conv.weight.data.shape)
conv.weight.data[:,:] = gaussian_filter
print(conv.weight.data.shape)

fig, ax = plt.subplots(1,1, figsize=(8,8))

with torch.no_grad():
    for batch, (x,y) in enumerate(loader):
        res = conv(x).cpu()[0, 0:3]
        print(res.shape)
        res = res.permute(1,2,0).numpy()
        res = np.cast["uint8"](res)
        ax.imshow(res)
        break

plt.show()
