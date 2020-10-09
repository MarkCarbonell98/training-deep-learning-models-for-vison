# -*- coding: utf-8 -*-
"""3_multi_layer_perceptron.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/constantinpape/training-deep-learning-models-for-vison/blob/master/day2/1_cnn.ipynb

<a href="https://colab.research.google.com/github/constantinpape/training-deep-learning-models-for-vison/blob/master/day2/1_cnn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# CNN on CIFAR10

In this exercise we will train our first convolutional neural network on the cifar 10 dataset.

## Preparation
"""

# Commented out IPython magic to ensure Python compatibility.
# load tensorboard extension
# %load_ext tensorboard

# import torch and other libraries
import os
import sys
sys.path.append(os.path.abspath('utils'))
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from cnn import CNN
import tqdm
import utils
import torch

# check if we have gpu support
# colab offers free gpus, however they are not activated by default.
# to activate the gpu, go to 'Runtime->Change runtime type'. 
# Then select 'GPU' in 'Hardware accelerator' and click 'Save'
have_gpu = torch.cuda.is_available()
# we need to define the device for torch, yadda yadda
if have_gpu:
    print("GPU is available")
    device = torch.device('cuda')
else:
    print("GPU is not available, training will run on the CPU")
    device = torch.device('cpu')


# we will reuse the training function, validation function and
# data preparation from the previous notebook

cifar_dir = './cifar10'
categories = os.listdir('./cifar10/train')
categories.sort()

# get training and validation data
train_dataset, val_dataset = utils.make_cifar_datasets(cifar_dir)

"""## CNN

Next we define the model architecture for our first CNN.
The network is made up of the following components:
- convolutional layers that convolve its input with a learnable filter, using less parameters than a fully connected layer while keeping spatial context
- max pooling that halves the image size
- fully connected layers at the end of the network to output a class prediction vector for the input

Note that both convolutional layers and pooling operations change the spatial
size of the data. You can find the formulas for this in the torch class descriptions: [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d), [nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).
See the comments in network definition below for an example.
"""

# instantiate the model
model = CNN(10)
model.to(device)

# Commented out IPython magic to ensure Python compatibility.
# instantiate loaders, loss, optimizer and tensorboard

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25)

optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

loss_function = torch.nn.NLLLoss()
loss_function.to(device)

tb_logger = torch.utils.tensorboard.SummaryWriter('runs/log_cnn_10_epochs')
# %tensorboard --logdir runs

n_epochs = 10
for epoch in tqdm.trange(n_epochs):
    utils.train(model, train_loader, loss_function, optimizer,
                device, epoch, tb_logger=tb_logger)
    step = (epoch + 1) * len(train_loader)
    utils.validate(model, val_loader, loss_function,
                   device, step,
                   tb_logger=tb_logger)



# evaluate the model on test data
test_dataset = utils.make_cifar_test_dataset(cifar_dir)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25)
predictions, labels = utils.validate(model, test_loader, loss_function,
                                     device, step=0, tb_logger=None)

print("Test accuracy:")
accuracy = sklearn.metrics.accuracy_score(labels, predictions)
print(accuracy)

fig, ax = plt.subplots(1, figsize=(8, 8))
utils.make_confusion_matrix(labels, predictions, categories, ax)

"""## Checkpoints and adaptive learning rate

The model weights can be saved in order to load the model again and run prediction in a different application. If in addiation the optimizer state is saved the model training can also be resumed from the same state.

Another important aspect of training neural networks is adapting the learning rate during training, which can increase model performance by converging to better optima. Here, we will condition the learning rate decrease on the validation accuracy using [torch.ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau).
"""

# functions to save and load a checkpoint (= serialized model and optimizer state +
# additional metadata)
# instantiate loaders, loss, optimizer and tensorboard

# instantiate the model
model = CNN(10)
model.to(device)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25)

optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

loss_function = torch.nn.NLLLoss()
loss_function.to(device)

tb_logger = torch.utils.tensorboard.SummaryWriter('runs/log_cnn1')

n_epochs = 15

# we use the best checkpoint as measured by the validation accuracy
best_accuracy = 0.
best_epoch = 0

# monitor the validation accuracy and decrease the learning rate when
# it starts to plateau
# NOTE: it's usually better to choose a higher patience value than a single epoch,
# we choose this value here in order to observe the changes when only training for 15 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                              mode='max',  # we evaluate based on accuracy, for which higher values are better
                              factor=0.5,  # half the learning rate
                              patience=1)  # number of epochs without improvement after which we reduce the lr
checkpoint_old_name = 'best_checkpoint.tar'
checkpoint_name = './best_checkpoint_reduce_on_plateau.tar'

for epoch in tqdm.trange(n_epochs):
    utils.train(model, train_loader, loss_function, optimizer,
                device, epoch, tb_logger=tb_logger)
    step = (epoch + 1) * len(train_loader)

    pred, labels = utils.validate(model, val_loader, loss_function,
                                  device, step,
                                  tb_logger=tb_logger)
    val_accuracy = sklearn.metrics.accuracy_score(labels, pred)
    scheduler.step(val_accuracy)
    
    # otherwise, check if this is our best epoch
    if val_accuracy > best_accuracy:
        # if it is, save this check point
        best_accuracy = val_accuracy
        best_epoch = epoch
        # print("Saving best checkpoint for epoch", epoch)
        utils.save_checkpoint(model, optimizer, epoch, checkpoint_name)

print("Best checkpoint is", best_epoch+1)

# load the model for the best checkpoint and evaluate it on the test data.
# how do the results compare to before?
model, _, _ = utils.load_checkpoint(checkpoint_name, model, optimizer)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25)
predictions, labels = utils.validate(model, test_loader, loss_function,
                                     device, step=0, tb_logger=None)

print("Test accuracy:")
accuracy = sklearn.metrics.accuracy_score(labels, predictions)
print(accuracy)

fig, ax = plt.subplots(1, figsize=(8, 8))
utils.make_confusion_matrix(labels, predictions, categories, ax)

"""## Visualize learned filters

The output of convolutional filters is still spatial, so it can be intepreted as images. Here, we inspect some of the filter responses of our network in order to visualize what features these filters pick up on.
"""

# apply the first convolutional filter of our model 
# to the first image from the training dataset
model.to(torch.device('cpu'))
model.eval()
im = train_dataset[0][0][None]

conv1_response = model.conv1(im).detach().numpy()[0]
print(conv1_response.shape)

# visualize the filters in the first layer
n_filters = 8
fig, axes = plt.subplots(1, 1 + n_filters, figsize=(16, 4))
im = im[0].numpy().transpose((1, 2, 0))
axes[0].imshow(im)
for chan_id in range(n_filters):
    axes[chan_id + 1].imshow(conv1_response[chan_id], cmap='gray')

plt.show()

"""## Tasks and Questions

Tasks:
- Construct a CNN that can be applied to input images of arbitrary size. Hint: Have a look at [nn.AdaptiveAveragePool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html).
- Train another network with more convolutional layers / poolings and observe how the performance changes.
- Visualize the convolutional filters for some of the deeper layers in your conv net.

Questions:
- How did the different models you have trained in this exercise perform? How do they compare to the non-convolutional models we have trained on the first day?
- How do you interpret the convolutional filters based on their visualisations? Can you see differences between the filters in the first and deeper layers?
"""

