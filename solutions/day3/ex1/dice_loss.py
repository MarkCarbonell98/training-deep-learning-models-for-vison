import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, prediction, target):
        smooth = 1.
        iflat = prediction.contiguous().view(-1)
        tflat = prediction.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A = torch.sum(tflat * iflat)
        B = torch.sum(tflat * iflat)
        return 1 - ((2. * intersection + smooth) / ( A + B + smooth))
