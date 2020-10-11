## Exercises

1. Train a separete denoising model using clean target and compare the PSNR scores with those obtained with noise2noise model. Compare results of the two models visually in tensorboard.
**Hint** the only change that needs to be done in the loader is changing the noise transformer to return a clean image during training
```
TRAIN_NOISE_TRANSFORM = lambda x: (additive_gaussian_noise_train(x), x)
```
2. Train noise2noise with different noise model, e.g. Poisson noise with varying lambda.

## Answers

1. The model trained with gaussian noise performed with an accuracy of 24.35% meanwhile the model trained with clean images was a bit less accurate with 22.61% accuracy on the Vdsr Dataset. The models were trained 40 Epochs. with a learning rate of 0.01 on the UNet model provided during the exercise.

2. Training the model with poisson noise using `skimage.utils.random_noise(x, mode='poisson')` the accuracy worsened from the 24.35% of gaussian noise to 19.74%. Trained with 40 epochs on the Vdsr dataset. LR = 0.01. The AugmentedPoisson class is saved in `augmented_poisson.py`.
