import torch
from torch import nn

def intersection_over_union(pred, label, eps = 1e-6):
    intersection = (pred & label).float().sum((2,3)) 
    union = (pred | label).float().sum((2,3))
    
    iou = (intersection + eps)/(union + eps)

    return iou[:, 1:].mean(), iou[:, 1:]

def precision_and_recall(pred, label, eps = 1e-6):
    intersect = (pred & label).float().sum((2,3))
    total_pixel_pred = pred.sum((2,3))
    total_pixel_truth = label.sum((2,3))
    precision = (intersect+eps)/(total_pixel_pred+eps)
    recall = (intersect+eps)/(total_pixel_truth+eps)
    return precision[:, 1:].mean(), precision[:, 1:], recall[:, 1:].mean(), recall[:, 1:]

def dice_score(pred, label, eps = 1e-6):
    intersect = (pred & label).float().sum((2,3))
    sum = pred.sum((2,3)) + label.sum((2,3))
    dice = (2*intersect+eps)/(sum+eps)
    return dice[:, 1:].mean(), dice[:, 1:]


class IoU(nn.Module):

    def __init__(self, threshold=0.5):

        super(IoU, self).__init__()

        self.threshold = threshold

 

    def forward(self, target, input):

        eps = 1e-10

        input_ = (input > self.threshold).data.float()

        target_ = (target > self.threshold).data.float()

 

        intersection = torch.clamp(input_ * target_, 0, 1)

        union = torch.clamp(input_ + target_, 0, 1)

 

        if torch.mean(intersection).lt(eps):

            return torch.Tensor([0., 0., 0., 0.])

        else:

            acc = torch.mean((input_ == target_).data.float())

            iou = torch.mean(intersection) / torch.mean(union)

            recall = torch.mean(intersection) / torch.mean(target_)

            precision = torch.mean(intersection) / torch.mean(input_)

            return torch.Tensor([acc, recall, precision, iou])