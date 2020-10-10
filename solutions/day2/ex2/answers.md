
## Tasks and Questions

Tasks:
- Implement one or two additional augmentations and train the model again using these. You can use [the torchvision transformations](https://pytorch.org/docs/stable/torchvision/transforms.html) for inspiration.

Questions:
- Compare the model results in this exercise.
- Can you think of any transformations that make use of symmetries/invariances not present here but present in other kinds of images (e.g. biomedical images)?

Advanced:
- Check out the other [normalization layers available in pytorch](https://pytorch.org/docs/stable/nn.html#normalization-layers). Which layers could be beneficial to BatchNorm here? Try training with them and see if this improves performance further.



## Task 1

- Two additional augmentations where implemented. One is called `random_center_crop` and the other `random_grayscale`. The correspondingly crop the center of the image given a tuple with two integers which demarks the center, and set the image's channels to grayscale given a certain probability.

### Answer 1
- Comparing the accuracy on the validation dataset. The SimpleCNN model which uses `random_flip` and `random_color_jitter` reached an accuracy of 64% while the CNNBatchNorm reached and accuracy of 69%. Using the new augmentations, the SimpleCNN reached 62% accuracy while the CNNBatchNorma 67%. Apparently the augmentations build during the exercise don't expose important image features like the ones originally shipped with the exercise.

### Answer 2
- The `random_flip` augmentation would not make much difference for biological images. Since cells can be positioned in any direction on the image. The `random_grayscale` augmentation would not be useful neither. Since microscopy images are usually already in black and which format. A transformation that could yield better results would be a laplacian filter, which demarks the cell's edges before the network learns on them. In this way it becomes easier to identify each cell on the image.

## Task 2

For this task I build a CNNBatchNormImproved which uses instance normalization in 2d before passing the input to the batch normalization function. Besides this I set the second batch normalization to BatchNorm1d instead of BatchNorm2d. This yieldes worst results than on Task 1. The CNNBatchNormImproved reached an accuracy of 56%. About 13% less than the performance of the original CNNBatchNorm which reached 69%.
