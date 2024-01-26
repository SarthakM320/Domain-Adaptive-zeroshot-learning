import torch
from torch import nn

def intersection_over_union(pred, label, eps = 1e-6):
    intersection = (pred & label).float().sum((2,3)) 
    union = (pred | label).float().sum((2,3))
    
    iou = (intersection + eps)/(union + eps)

    return iou.mean(), iou

def precision_and_recall(pred, label, eps = 1e-6):
    intersect = (pred & label).float().sum((2,3))
    total_pixel_pred = pred.sum((2,3))
    total_pixel_truth = label.sum((2,3))
    precision = (intersect+eps)/(total_pixel_pred+eps)
    recall = (intersect+eps)/(total_pixel_truth+eps)
    return precision.mean(), precision, recall.mean(), recall

def dice_score(pred, label, eps = 1e-6):
    intersect = (pred & label).float().sum((2,3))
    sum = pred.sum((2,3)) + label.sum((2,3))
    dice = (2*intersect+eps)/(sum+eps)
    return dice.mean(), dice[0]