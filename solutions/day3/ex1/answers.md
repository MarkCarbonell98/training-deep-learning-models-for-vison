## Task:
Use `ls` to explore the contents of both folders. Running `ls your_folder_name` should display you what is stored in the folder of your interest.

 How are the images stored? What format do they have? What about the ground truth (the annotation masks)? Which format are they stored in?

Hint: you can use the following function to display the images:

## Additional Exercises 

1. Implement and compare at least 2 of the following architecture variants of the U-Net:
    * use [Dropout](https://pytorch.org/docs/stable/nn.html#dropout-layers) in the decoder path
    * use [BatchNorm](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d) to normalize layer inputs
    * use [GroupNorm](https://pytorch.org/docs/stable/nn.html#torch.nn.GroupNorm) to normalize convolutional group inputs
    * use [ELU-Activations](https://pytorch.org/docs/stable/nn.html#torch.nn.ELU) instead of ReLU-Activations

2. Use the Dice coefficient as loss function. Before we only used it for validation, but it is differentiable and can thus also be used as loss. Compare to the results from exercise 2. 

Hi# implementnt: The optimizer we use finds minima of the loss, but the minimal value for the Dice coefficient corresponds to a bad segmentation. How do we need to change the Dice coefficient to use it as loss nonetheless?

3. Add one more layer to the Unet model (currently it has 4). Compare the results.

## Advanced Exercises

1. Visualize the graph (model) that we are using with TensorBoard

2. Write your own data transform (e.g., RandomRotate)


## Answers

1. The BatchNorm and ELU-Activations variants were chosen and trained for 15 epochs. The accuracy reached by  the BatchNorm variant in `unet_batchnorm.py` was 42%. The one reached by the ELU variant in `unet_elu.py` was 26%. Apparently using batch normalization has a greater effect on performance than using ELU activation layers. The original model reached 7% accuracy with the same amount of epochs.

2. By training the networks using the DiceCoefficient class as a loss function, results were pretty bad. The variantion of the UNet using batchnorm reached 6% accuracy, while the original and the ELU version only 1% for 15 epochs.

- In order to convert the dice coefficients to a loss function I wrote a `DiceLoss` class in `dice_loss.py` which applies the same principle of the Dice Coefficients, but smooths the final output by a factor of 1 for stability and substracts the result from 1. All three models performed better using `DiceLoss` than `DiceCoefficients` as a loss function. BatchNorm variant reached 47% accuracy, ELU variant 35% and the original UNet 0%.

3. The UNet with one more layer is located in `unet_five_layers.py`. My tests went wrong in this part since I got 0% `val_metric` for all three loss functions (MSE, DiceCoeffs and DiceLoss)

## Advanced Answers

1. The graphs are saved in the `./runs` directory of this repository. Feel free to check them out using tensorboard.

2. The implementatoin of the data transform `RandomRotate` can be found under `random_rotate.py`. It is implemented in a class similar to RandomCrop.


