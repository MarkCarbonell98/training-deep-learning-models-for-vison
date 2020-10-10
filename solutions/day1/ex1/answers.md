
## Torchvision

Note: there is torch library for computer vision: [torchvision](https://pytorch.org/docs/stable/torchvision/index.html). It has many datasets, transformations etc. already. For example it also has a [prefab cifar dataset](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar).

We do not use torchvision here for two reasons:
- to show how to implement a torch.dataset yourself, so that you can implement it for new data you are working with
- torchvision uses [PIL](https://pillow.readthedocs.io/en/stable/) to represent images, which is rather outdated

Still, torchvision contains helpful functionality and many datasets, so it's a very useful library.

## Tasks

- Apply more advanced transforms in the dataset: for example you could blur the images or rotate them on the fly.

## Answers 

## Task 1
- For this exercise I implemented an `apply_and_show_transforms` functions for quickly visualizing the results of each transform. The implemented transformations are

1. rotate90
2. rotate180
3. rotate270
4. blur
5. edge - applies the laplacian edge filter on the images
6. red
7. green
8. blue
9. crop - crops the 32 * 32 image to a random size while preserving original dimensions


