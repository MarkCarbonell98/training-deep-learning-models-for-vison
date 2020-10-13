import torch
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from constants import (
    PASCAL_CLASSES,
    COLORS,
    COCO_CLASSES
)



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, device='cpu'):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

def detect(im, model, transform, device='cpu'):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    img = img.to(device)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
    return probas[keep], bboxes_scaled


def show_random_dataset_image(dataset, idx=-1):
    if idx == -1:
        idx = np.random.randint(0, len(dataset))    # take a random sample
    img, mask = dataset[idx]                    # get the image and the nuclei masks
    img, mask = np.array(img), np.array(mask)
    f, axarr = plt.subplots(1, 2)               # make two plots on one figure
    axarr[0].imshow(img)                     # show the image
    axarr[1].imshow(mask)                    # show the masks
    _ = [ax.axis('off') for ax in axarr]        # remove the axes
    print(f'Image size: {img.shape}')


def bbox(img, label):
    """
    Extracts the bounding box of a given label in the image.

    Returns:
        tuple (xmin, ymin, xmax, ymax): coordinated of the top left and bottom right corners of the bounding box
    """
    a = np.where(img == label)
    bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]) , np.max(a[0])
    return bbox

def find_boxes(label_img, min_size=100):
    """
    Given the labeled image 'label_img', finds the bounding boxes around different connected components (instances)
    in the image.
    
    Returns:
        tuple (classes, bounding_boxes): where classes is an array of strings, i.e. name of the class of object int the bounding box,
                                         bounding_boxes is an array of tuples of the form (xmin, ymin, ymax, ymax).
    """
    label_img = np.array(label_img)
    label_copy = label_img.copy()
    # zero-out instances boundaries, which in Pascal VOC are given the value of 255
    label_copy[label_copy == 255] = 0
    # find connected components in the labeled image
    connected_components = measure.label(label_copy,  connectivity=1)
    
    classes = []
    boxes = []
    labels, counts = np.unique(connected_components, return_counts=True)
    # iterate over instances and get the bounding box and the corresponding class
    for label, count in zip(labels, counts):
        # skip 0-label or instances smaller than 'min_size' pixels
        # if label == 0 or count < min_size:
        #     continue

        if label != 0 and count >= min_size:
            # extract and save the bounding box
            boxes.append(bbox(connected_components, label))

            # get the class of the object with a given label
            c, n = np.unique(label_img[connected_components == label], return_counts=True)
            ind = c[n.argmax()]
            classes.append(PASCAL_CLASSES[ind])
        
    
    return classes, boxes

def plot_box(pil_img, classes, boxes, ax):
    ax.imshow(pil_img)
    for cl, (xmin, ymin, xmax, ymax), c in zip(classes, boxes, COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        ax.text(xmin, ymin, cl, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.axis('off')

def plot_demo_row(pil_img, gt_classes, pred_classes, gt_boxes, pred_boxes, ax, ax2):
    plot_box(pil_img, gt_classes, gt_boxes, ax)
    ax.set_title("Predictions")
    plot_box(pil_img, pred_classes, pred_boxes, ax2)
    ax2.set_title("Ground Truth")


def prob_to_classes(prob):
    return [COCO_CLASSES[p.argmax()] for p in prob]
    
def plot_results(pil_img, prob, boxes):
    classes = prob_to_classes(prob)
    plot_boxes(pil_img, classes, boxes)

def plot_boxes(pil_img, classes, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for cl, (xmin, ymin, xmax, ymax), c in zip(classes, boxes, COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        
        ax.text(xmin, ymin, cl, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        
    plt.axis('off')

