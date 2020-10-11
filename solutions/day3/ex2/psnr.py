
import os
import sys
sys.path.append(os.path.abspath('utils'))
import utils
from skimage.metrics import peak_signal_noise_ratio

class PSNR:
    def __call__(self, image_true, image_test):
        image_true = utils.clip_to_uint8(image_true)
        image_test = utils.clip_to_uint8(image_test)
        image_true = image_true.detach().cpu().numpy()
        image_test = image_test.detach().cpu().numpy()
        return peak_signal_noise_ratio(image_true, image_test)
    
