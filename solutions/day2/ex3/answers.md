
## Tasks and Questions

Tasks:
- Read up on some of the models in [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html) and train at least one of them on this data.
- Combine the best performing model in this exercise with data augmentation (previous exercise).

Questions:
- What's the best accuracy you have achieved on CIFAR10 over all the exercises? Which model and training procedure did lead to it?
- What would your next steps be to improve this performance?
- Do you think the performance possible on cifar will improve significantly with much larger models (= models with a lot more parameters)?

## Answers

### Task 1:
-  The chosen model was `resnet34`. In comparison with `resnet18` which accomplished 65% accuracy on the CIFAR10 dataset, `resnet34` reached %%

### Task 2:
-  The model that performed the best was ... with ... accuracy. The data augmentations used where the `random_flip` and `random_color_jitter` transformations. These transformations improved performance on the CNNBatchNorm of the previous exercise, therefore it was decided to augment this model's data with them.

### Question 1:
- The best accuracy reached so far was ... by the model ... with augmentations ...


### Question 2:
-

### Question 3:
- Not necessarily. The cifar dataset's images are only 32 pixels wide. Their resolution is very low. Therefore there is a limit for the amount of significant parameters you can effectively train to reach a better performance. However, training a large network like `vgg16` for more than 100 epochs on cifar will probably yield better results than a smaller model. Therefore the conclusion is yes, performance can improve significantly with more parameters, but not necessarilly. It will depend mainly on the significance of the parameters used, not the amount of them.





