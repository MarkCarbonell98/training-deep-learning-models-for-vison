# -*- coding: utf-8 -*-
"""Kopie von Copy of ReadingData.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JgTMhR4_BdwhcRjSCs3t3rGjQHerSrbN

# Data preparation for a classification model

In our first exercise, we will download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60,0000 small images (32 x 32 pixel) which fall into 10 different classes (automobile, airplane, ...).

Here, we will learn how to visualize the data and prepare it for training a classifier model.

## Import python libraries
"""

import os
import sys
sys.path.append(os.path.abspath('utils'))
import utils
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from tqdm import tqdm
import skimage.filters as filters
from torch.utils.data import Dataset
from functools import partial  # to bind function arguments
import torch



"""## Download and read the data

We download and load the images and corresponding labels to inspect them (and later use it to train a classifier).

The data is organised in a folder structure as follows:

```
train/
  airplane/
    001.png
    002.png
    ...
  automobile/
    001.png
    002.png
    ...
```

and similarly for the test data.
"""

cifar_dir = './cifar10'

# first, list the categories available in the data
data_dir = os.path.join(cifar_dir, "train")
categories = os.listdir(data_dir)
categories.sort()
print(categories)

# next load the images and labels
images = []
labels = []
for label_id, category in tqdm(enumerate(categories), total=len(categories)):
    category_dir = os.path.join(data_dir, category)
    image_names = os.listdir(category_dir)
    for im_name in image_names:
        im_file = os.path.join(category_dir, im_name)
        images.append(imread(im_file))
        labels.append(label_id)

# make numpy arrays out of the lists
# for th images, we stack along a new first axis
images = np.concatenate([im[None] for im in images], axis=0)
labels = np.array(labels)

print("Number of images:", len(images))
print("Number of labels:", len(labels))

# plot one image for each category
fig, ax = plt.subplots(1, 10, figsize=(18, 6))
label_list = labels.tolist()
for label_id, category in enumerate(categories):
    ax[label_id].imshow(images[label_list.index(label_id)])
    ax[label_id].set_title(category)
plt.show()

"""## Convolutional Filters

As a first step we apply some simple filters on the images.
In particular, we use convolutional filters, that can be expressed as convolution of a kernel with the image, which will be important for the concept of Convolutional Neural Networks that we will introduce later.

For now, we will use some filter available in [skimage.filters](https://scikit-image.org/docs/dev/api/skimage.filters.html).
"""

image = images[0]
filtered_gaussian = filters.gaussian(image, sigma=1., multichannel=True)
filtered_laplacian = filters.laplace(image)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(image)
ax[1].imshow(filtered_gaussian)
ax[2].imshow(filtered_laplacian)

"""## Preparation for pytorch

In order to use the CIFAR data to with pytorch, we need to transform the data into the compatible data structures. In particular, pytorch expects all numerical data as [torch.tensor](https://pytorch.org/docs/stable/tensors.html).
To provide the data as tensors, we will wrap them in a [torch.dataset](https://pytorch.org/docs/stable/data.html) and implement a mechanism to apply transformations to the data on the fly. We will use these transformations to bring the data into a format that pytorch can ingest and later also use them for other purposes such as data augmentation.
"""

# datasets have to be sub-classes from torch.util.data.Dataset

# what transofrmations do we need to feed this data to pytorch?

# first, let's check the shape of our images:
print(image.shape)

# as we see, the images are stored in the order width, height, channel (WHC), 
# i.e. the first two axes are the image axes and the last axis
# corresponds to the color channel.
# pytorch however expects the color channel as first axis, i.e. CWH.
# so our first transform switches the chanels

# note that we have implemented the dataset in such a way, that the transforms
# are functions that take bot the data (or image) and target as parameters.
# thus we here accept the target (which is just the class label for the image) 
# as second parameter and return it without changing it
def to_channel_first(image, target):
    """ Transform images with color channel last (WHC) to channel first (CWH)
    """
    # put channel first
    image = image.transpose((2, 0, 1)) # sets the last axis to be the first one, first axis to be the second one, and second axis as the last one
    return image, target

# next, let's see what datatype and value range our images have
label = label_list[0]
print(image.dtype)
print(image.min(), image.max())
print(image.shape)
transformed_image, _ = to_channel_first(image, label)
print(transformed_image.shape)
print(image.min(axis=(1,2)).shape) # returns 1D list with the min values
print(image.min(axis=(1,2), keepdims=True).shape) # returns the same list buth keeping the dimensions of the original image

# as we can see, the images are stored as 8 bit integers with a value range [0, 255]
# instead, torch expects images as 32 bit floats that should also be normalized to a 'reasonable' data range.
# here, we normalize the image such that all channels are in range 0 to 1
def normalize(image, target, channel_wise=True):
    eps = 1.e-6
    image = image.astype('float32')
    chan_min = image.min(axis=(1, 2), keepdims=True)
    image -= chan_min
    chan_max = image.max(axis=(1, 2), keepdims=True)
    image /= (chan_max + eps)
    return image, target


# finally, we need to transform the input from a numpy array to a torch tensor
# and also return the target (which in our case is a scalar) as a tensor
def to_tensor(image, target):
    return torch.from_numpy(image), torch.tensor([target], dtype=torch.int64)


# we also need a way to apply multiple transforms
# (note that alternatively we could also have accepted a list of transforms
# in DatasetWithTransform)
def compose(image, target, transforms):
    for trafo in transforms:
        image, target = trafo(image, target)
    return image, target

# create the dataset with the transformations

trafos = [to_channel_first, normalize, to_tensor]
trafo = partial(compose, transforms=trafos) # freezes compose with trafos as a function call

dataset = utils.DatasetWithTransform(images, labels, transform=trafo)
print(len(dataset))

# function to show an image target pair returned from the dataset
def show_image(ax, image, target, trafo_name):
    # need to go back to numpy array and WHC axis order
    image = image.numpy().transpose((1, 2, 0))
    # find the label name
    label = categories[target.item()]
    ax.imshow(image)
    ax.set_title("%s : %s".format(label, trafo_name))

def show_sample_images(dataset, n_samples, ids):
    n = len(dataset)
    fig, ax = plt.subplots(1, n_samples, figsize=(18,4))
    for sample, id in enumerate(ids):
        id = id[0]
        img, target = dataset[id]
        trafo_name = dataset.transform.keywords['transforms'][-1].__name__

        assert np.isclose(image.min(), 0.)
        assert np.isclose(image.max(), 1.)

        show_image(ax[sample], img, target, trafo_name)


# sample a few images from the dataset and check their label

def rotate90(image, target):
    assert isinstance(image, torch.Tensor)
    return image.transpose(1,2), target

def rotate180(image, target):
    assert isinstance(image, torch.Tensor)
    return image.flip(1), target

def rotate270(image, target):
    assert isinstance(image, torch.Tensor)
    return image.transpose(1,2).flip(1), target

def blur(image, target):
    return torch.from_numpy(filters.gaussian(image, sigma=1., multichannel=True)), target

def edge(image, target):
    image = torch.from_numpy(filters.laplace(image, ksize=3, mask=None))
    return image, target

def red(image, target):
    image[1:3, ] = 0
    return image, target

def green(image, target):
    image[0:3:2, ] = 0
    return image, target

def blue(image, target):
    image[0:2, ] = 0
    return image, target

def crop(image, target):
    n = image.shape[1]
    lim = np.random.randint(5, n)
    image = image[..., lim:n, lim:n]
    return image, target



def apply_and_show_transforms(transforms):
    ids = np.random.randint(0, 50000, (8,1)).tolist()
    for new_trafo in transforms:
        all_trafos = [to_channel_first, normalize, to_tensor, new_trafo]
        trafo = partial(compose, transforms=all_trafos)
        dataset = utils.DatasetWithTransform(images, labels, transform=trafo)
        show_sample_images(dataset, 8, ids)

apply_and_show_transforms([rotate90, rotate180, rotate270, blur, edge, red, green, blue, crop])

"""## Torchvision

Note: there is torch library for computer vision: [torchvision](https://pytorch.org/docs/stable/torchvision/index.html). It has many datasets, transformations etc. already. For example it also has a [prefab cifar dataset](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar).

We do not use torchvision here for two reasons:
- to show how to implement a torch.dataset yourself, so that you can implement it for new data you are working with
- torchvision uses [PIL](https://pillow.readthedocs.io/en/stable/) to represent images, which is rather outdated

Still, torchvision contains helpful functionality and many datasets, so it's a very useful library.

## Tasks

- Aply more advanced transforms in the dataset: for example you could blur the images or rotate them on the fly.
"""
