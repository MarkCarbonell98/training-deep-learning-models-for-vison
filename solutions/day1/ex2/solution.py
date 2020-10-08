import os
import sys

sys.path.append(os.path.abspath('utils'))
print(sys.path[-1])

import subprocess
import numpy as np
import torch
import sklearn
import skimage
import utils
from logistic_regressor_conv_filters import LogisticRegressor
from functools import partial
import tqdm
import matplotlib.pyplot as plt
import tensorboard

tb = tensorboard.program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'runs/logistic_regressor_conv_filters'])
url = tb.launch()
print('TensorBoard running on: ', url) 


# bashCommand = "tensorboard --logdir runs"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# out, err = process.communicate()
# print("Process output: ", out)
# print("Process error: ", err)

is_gpu = torch.cuda.is_available()

if is_gpu:
    print("GPU Available")
    device = torch.device('cuda')
else:
    print("GPU not available")
    device = torch.device('cpu')

categories = os.listdir('./cifar10/train')
categories.sort()
print(categories)

cifar_dir = './cifar10'

images, labels = utils.load_cifar(os.path.join(cifar_dir, 'train'))

n_imgs = len(images)

# train_test_split from sklearn shuffles and stratifies the data such that
# the same number of samples per classes is present in train and val splits

(train_imgs, val_imgs, train_labels, val_labels) = sklearn.model_selection.train_test_split(images, labels, shuffle=True, test_size=.15, stratify=labels)

assert len(train_imgs) == len(train_labels)
assert len(val_imgs) == len(val_labels)
assert len(train_imgs) + len(val_imgs) == n_imgs

trafos = [utils.to_channel_first, utils.normalize, utils.to_tensor]
trafos = partial(utils.compose, transforms=trafos)

train_data = utils.DatasetWithTransform(train_imgs, train_labels, transform=trafos)
val_data = utils.DatasetWithTransform(val_imgs, val_labels, transform=trafos)

print("N Training: ", len(train_imgs))
print("N Val: ", len(val_imgs))

n_pixels = 9 * 32 * 32
n_classes = 10
model = LogisticRegressor(n_pixels, n_classes)
model.to(device)

train_batch_size = 4
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

val_batch_size= 25
val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)
loss = torch.nn.NLLLoss()
loss.to(device)

tb_logger = torch.utils.tensorboard.SummaryWriter('runs/log_reg_gauss_laplace_skimage')

for epoch in tqdm.trange(4):
    utils.train(model, train_loader, loss, optimizer, device, epoch, tb_logger=tb_logger)
    step = (epoch + 1) * len(train_loader)
    utils.validate(model, val_loader, loss, device, step, tb_logger=tb_logger)

test_imgs, test_labels = utils.load_cifar(os.path.join(cifar_dir, 'test'))

test_data = utils.DatasetWithTransform(test_imgs, test_labels, transform = trafos)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size)

pred, test_labels = utils.validate(model, test_loader, loss, device, 0)
accuracy = sklearn.metrics.accuracy_score(test_labels, pred)
print("Test Accuracy: ", accuracy)

fig, ax = plt.subplots(1, figsize=(8,8))
utils.make_confusion_matrix(test_labels, pred, categories, ax)

plt.show()



