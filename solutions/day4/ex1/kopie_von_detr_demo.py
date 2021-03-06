# *- coding: utf-8 -*-
"""Kopie von detr_demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EXoF72eAPCy6wrokSaUbERJIoyzxon0z

<a href="https://colab.research.google.com/github/constantinpape/training-deep-learning-models-for-vison/blob/master/day4/detr_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Object Detection with DETR - a minimal implementation

Based on the [notebooks from Facebook Research](https://github.com/facebookresearch/detr#notebooks)

In this notebook we show a demo of DETR (Detection Transformer), with slight differences with the baseline model in the paper.

We show how to define the model, load pretrained weights and visualize bounding box and class predictions.

Let's start with some common imports.
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import zipfile
import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import numpy as np
from skimage import measure
import tqdm

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
from detr import DETRdemo
from detr_utils import *
torch.set_grad_enabled(False);
from constants import (
    PASCAL_CLASSES,
    COLORS,
    COCO_CLASSES
)
from sklearn.metrics import average_precision_score

"""## DETR
Here is a minimal implementation of DETR:
"""

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device('cuda')
else:
    print("GPU is not available, training will run on the CPU")
    device = torch.device('cpu')

"""As you can see, DETR architecture is very simple, thanks to the representational power of the Transformer. There are two main components:
* a convolutional backbone - we use ResNet-50 in this demo
* a Transformer - we use the default PyTorch nn.Transformer

Let's construct the model with 80 COCO output classes + 1 ⦰ "no object" class and load the pretrained weights.
The weights are saved in half precision to save bandwidth without hurting model accuracy.
"""

detr = DETRdemo(num_classes=91)

state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)

detr.load_state_dict(state_dict)
detr.eval()

detr = detr.to(device)

"""## Computing predictions with DETR

The pre-trained DETR model that we have just loaded has been trained on the 80 COCO classes, with class indices ranging from 1 to 90 (that's why we considered 91 classes in the model construction).
In the following cells, we define the mapping from class indices to names.
"""

# COCO classes
# colors for visualization
"""DETR uses standard ImageNet normalization, and output boxes in relative image coordinates in $[x_{\text{center}}, y_{\text{center}}, w, h]$ format, where $[x_{\text{center}}, y_{\text{center}}]$ is the predicted center of the bounding box, and $w, h$ its width and height. Because the coordinates are relative to the image dimension and lies between $[0, 1]$, we convert predictions to absolute image coordinates and $[x_0, y_0, x_1, y_1]$ format for visualization purposes."""

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(500),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing

"""Let's put everything together in a `detect` function:"""


"""## Loading Pascal VOC 2012 dataset
Before we start let's download the Pascal VOC validation set from the [here](https://oc.embl.de/index.php/s/bkBUhSajTPP0lUP) and save it in your Google Drive. The archive is 2GB in size so it will take a while.

After the ZIP file has been successfully uploaded to your Google Drive, mount your Drive following [the instructions](https://colab.research.google.com/github/constantinpape/training-deep-learning-models-for-vison/blob/master/exercises/mount-gdrive-in-colab.ipynb) and unzip the archive.
"""

"""Let's create the Pascal VOC loader from `torchvision` package and show some images with the ground truth segmentation masks."""

root_dir = "./PascalVOC2012"

voc_dataset = VOCSegmentation(root_dir, year='2012', image_set='trainval', download=False)

"""Before we move on let's define the 20 classes of objects avialable in the Pascal VOC dataset"""

# Pascal VOC classes, modifed to match the COCO classes, i.e. the following 4 class names were mapped:
# aeroplane -> airplane
# diningtable -> dining table
# motorbike -> motorcycle
# sofa -> couch
# tvmonitor -> tv

"""For the exercises we will need a helper function which extracts the bounding boxes around the individual instances given the ground truth semantic mask."""


"""Visualize the bounding boxes on a given image from the Pascal VOC dataset"""

indexes = torch.randint(0, len(voc_dataset), (20,))
for index, i in enumerate(indexes):
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    img, label_img = voc_dataset[i]
    gt_classes, gt_boxes = find_boxes(label_img)
    # plot_boxes(img, classes, boxes)
    scores, pred_boxes = detect(img, detr, transform, device=device)
    # plot_results(img, scores, boxes)
    pred_classes = prob_to_classes(scores)
    plot_demo_row(img, gt_classes, pred_classes, gt_boxes, pred_boxes, ax1, ax2)

# plt.show()
    
"""## Using DETR
Try DETRdemo model on the same image you've chosen above.
"""

"""Let's now visualize the model predictions"""

"""## Excercises

1. Use the pre-trained DETR model to detect objects in 20 images randomly sampled from the Pascal VOC dataset. Show predicted bounding boxes together with the ground truth bouding boxes (see `find_boxes` function) side by side. Quantitatively how does the model perform? Do you see any irregularities between the predicted bounding boxes and the ground truth masks?


2. Quantify the object detection performance on the Pascal VOC 2012 dataset using the Mean Average Precision metric. Given a function which returns the ground truth bounding boxes together with their corresponding classes (see `find_boxes`) and bounding box predictions given by the DETR model, compute the `mAP` score at different IoU thresholds (e.g. 0.4, 0.5, 0.75) on the **entire** Pascal VOC 2012 dataset. Details of how to implement `mAP` for object detection ca be found e.g. [here](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).
You can use the pseudo code below as a starting point

**Hint**
bear in mind that COCO dataset contains 81 classes, whereas Pascal VOC contains 20 classes. For images where DETR model returns one of the 61 classes not present in the Pascal VOC, simply ignore the predicted bounding box instead of counting it as a False Positive.
"""

def calc_iou(gt_box, pred_box):
    xmin = max(gt_box[0], pred_box[0])
    ymin = max(gt_box[1], pred_box[1])
    xmax = min(gt_box[2], pred_box[2])
    ymax = min(gt_box[3], pred_box[3])

    intersect = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    iou = intersect / float(gt_area - pred_area - intersect)
    return iou

def average_precision(gt_classes, gt_boxes, pred_classes, pred_boxes, iou_threshold):
    y_true = []
    y_scores = []
    for gt_c, gt_b in zip(gt_classes, gt_boxes):
        for pred_c, pred_b in zip(pred_classes, pred_boxes):
            if gt_c == pred_c:
                iou = calc_iou(gt_b, pred_b)
                if iou >= iou_threshold:
                    y_true.append(1)
                    y_scores.append(iou)
                else:
                    y_true.append(0)
                    y_scores.append(iou)
    y_true = [0] if len(y_true) == 0 else y_true
    y_scores = [0] if len(y_scores) == 0 else y_true
    ap = average_precision_score(np.array(y_true), np.array(y_scores))
    if math.isnan(ap):
        ap = min(y_scores)
    return ap


# pick an Intersection over Union threshold


average_precision_list = []

print("VOC Dataset length: ", len(voc_dataset))
# iterate directly over the Dataset
for iou_threshold in [.4, .5, .75]:
    print(f"Calculating mAP with IoU threshold: {iou_threshold}")
    for img, label in tqdm.tqdm(voc_dataset, total=len(voc_dataset)):
        img_t = transform(img).unsqueeze(0)
        img_t = img_t.to(device)
        if img_t.shape[-2] <= 1600 and img_t.shape[-1] <= 1600:
            # extact ground truth classes and ground truth boxes from the labeled image
            gt_classes, gt_boxes = find_boxes(label)
            # run the prediction with DETR
            pred_prob, pred_boxes = detect(img, detr, transform, device=device)
            
            pred_classes = prob_to_classes(pred_prob)
            
            ap = average_precision(gt_classes, gt_boxes, pred_classes, pred_boxes, iou_threshold)
            
            average_precision_list.append(ap)
    print(f'mAP@{iou_threshold}:', np.mean(average_precision_list))

plt.show()
