# Tasks and Questions:
## Tasks:

- Train a model with more preset filters and compare the different models via tensorboard and on the test dataset.

# Questions:


- What accuracy do the different models reach?
1. First Model with No filters = 0.2926
2. Second Model with gaussian and laplacian filters = 0.319
3. Third Model with median and robets filters = 0.2969
4. Fourth Model with gaussian and laplacian conv filters = X


- Which accuracy do you expect by guessing?
1. A maximum of 10% accuracy. Since each class would have the same probability for each prediction. Therefore it's expected that every class would be predicted the same amount of times for different data.


- Can you find any systematic errors from the confusion matrix?
1. The predictions made for a single class are much more higher than for any other class. The columns where the prediction with the largest confidence is made is the one which accumulates the most values.

-  Advanced:
The filters we have used here can be expressed as convolutions.
Express the gaussian filter using nn.Conv2d and train a model using these filters.
Can you also explace the laplace filter?

1. Yes, I implemented both filters as convolutional layers in the `logistic_regressor_conv_layers.py`
