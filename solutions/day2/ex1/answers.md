## Tasks and Questions

Tasks:
- Construct a CNN that can be applied to input images of arbitrary size. Hint: Have a look at [nn.AdaptiveAveragePool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html).
- Train another network with more convolutional layers / poolings and observe how the performance changes.
- Visualize the convolutional filters for some of the deeper layers in your conv net.

Questions:
- How did the different models you have trained in this exercise perform? How do they compare to the non-convolutional models we have trained on the first day?
- How do you interpret the convolutional filters based on their visualisations? Can you see differences between the filters in the first and deeper layers?

## Answers

- The original CNN performed with an accuracy of 50%. Compared to the non-convolutional models like the logistic regressor this is an overall improvement of 10% in overall accuracy
- The CNN with AdaptiveAvgPool2d worked improved the performance of the original CNN by a10%. Reaching 60% accuracy
- The CNN with two extra convolutional layers, and corresponding 2d MaxPoolings performed 4% better than the model with only the AdaptiveAvgPool2d. Reaching 64% accuracy

- The convolutional filters of the first convolutional layers seem to be quite similar to the original image. In comparison with deeper filters (Up from the third convolutional layer) the filter images start to look more like patterns extracted from the image. The deeper the filter the more difficult it becomes to say what the original image looks like given the filter image.

