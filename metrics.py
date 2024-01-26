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