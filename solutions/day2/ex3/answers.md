
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
-  The chosen model was `resnet34`. In comparison with `resnet18` which accomplished 65% accuracy on the CIFAR10 dataset, `resnet34` reached 28,9%

### Task 2:
-  The model that performed the best was `resnet18` with 65% accuracy. The data augmentations used where the `random_flip` and `random_color_jitter` transformations. These transformations improved performance on the CNNBatchNorm of the previous exercise, therefore it was decided to augment this model's data with them. Interestingly, the augmentations decreased the accuracy of ResNet18 to 54% on the same data, with the same amount of epochs.

### Question 1:
- The best accuracy reached so far was 69% by the model CNNBatchNorm from `day2/` with augmentations `random_flip` and `random_color_jitter` data augmentations.

### Question 2:
- The steps to improve performance would be:
1. Train a much larger number of epochs
2. Enforce reduction of learning on plateau for the model's optimizer
3. Try different data augmentations which improve the accuracy of the model
4. Use a pretrained pytorch model trained on ImageNet and then finetune it on the Cifar dataset

### Question 3:
- Not necessarily. The cifar dataset's images are only 32 pixels wide. Their resolution is very low. Therefore there is a limit for the amount of significant parameters you can effectively train to reach a better performance. However, training a large network like `vgg16` for more than 100 epochs on cifar will probably yield better results than a smaller model. Therefore the conclusion is yes, performance can improve significantly with more parameters, but not necessarilly. It will depend mainly on the significance of the parameters used, not the amount of them.





