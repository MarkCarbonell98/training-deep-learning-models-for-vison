
## Excercises

1. Use the pre-trained DETR model to detect objects in 20 images randomly sampled from the Pascal VOC dataset. Show predicted bounding boxes together with the ground truth bouding boxes (see `find_boxes` function) side by side. Quantitatively how does the model perform? Do you see any irregularities between the predicted bounding boxes and the ground truth masks?


2. Quantify the object detection performance on the Pascal VOC 2012 dataset using the Mean Average Precision metric. Given a function which returns the ground truth bounding boxes together with their corresponding classes (see `find_boxes`) and bounding box predictions given by the DETR model, compute the `mAP` score at different IoU thresholds (e.g. 0.4, 0.5, 0.75) on the **entire** Pascal VOC 2012 dataset. Details of how to implement `mAP` for object detection ca be found e.g. [here](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).
You can use the pseudo code below as a starting point

**Hint**
bear in mind that COCO dataset contains 81 classes, whereas Pascal VOC contains 20 classes. For images where DETR model returns one of the 61 classes not present in the Pascal VOC, simply ignore the predicted bounding box instead of counting it as a False Positive.

## Answers

1. The DETR tends to predict several bounding boxes of the same class around the object where a single bounding box should suffice. It has serious problems prediction books. And the dimension of the bounding boxes are not very accurate. Sometimes persons are only halfway into the predicted bounding box whereas in the GT they are completely included in the bounding box.

2. The DETR model performed with 14.2% mAP on the `trainval` Pascal VOC 2012 dataset with an IoU threshold of 40%. The results came in descending order with higher IoU thresholds. Ranking 13.6% with IoU threshold of 50% and 12.6%  with the 75%. Interestingly, I found a bug in the `sklearn` implementation of the `average_precision` function which is documented [here](https://github.com/scikit-learn/scikit-learn/issues/8245) where the function returns a NaN value if the label list is composed of purely negative labels. 
