import numpy as np


class RandomRotate():
    """Rotate an image randomly"""
    def __init__(self, k = np.random.randint(0, 4)):
        # check if the crop size is of a valid type
        if isinstance(k, int) and k in range(0,4):
            self.k = k
        else:
            print("Input must be a random integer between 0 and 4")

    # this function makes our class callable 
    def __call__(self, sample):
        # we need to crop both input and mask at the same time
        assert len(sample) == 2
        image, mask = sample
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
        # the first dimension is channels, then width, then height
        return image, mask

