import torch
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class VdsrDataset(Dataset):
    def __init__(self, root_dir, noise_transform, crop_size=256):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(root_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        assert noise_transform is not None
        # transforms the image according to the noise model
        self.noise_transform = noise_transform
        #  standard transformations to apply to the input images
        self.inp_transforms = transforms.Compose([
            # randomly crop the image, paddig if necessary
            transforms.RandomCrop(crop_size, pad_if_needed=True, padding_mode='reflect'),
            # converts numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            transforms.ToTensor()
        ])
        
    # get the total number of samples
    def __len__(self):
        return len(self.image_paths)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # read image from disk
        img = imageio.imread(self.image_paths[idx])
        # convert signle-channel images
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        # convet to PIL image
        img = Image.fromarray(img)
        # apply standard augmentations
        img = self.inp_transforms(img)
        # convert [0, 1] to [-0.5, 0.5]
        img = img - 0.5
        # apply the noise model and return a source and target image
        return self.noise_transform(img)

