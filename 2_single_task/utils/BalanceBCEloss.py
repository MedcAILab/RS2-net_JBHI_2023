import torch
import torch.nn.functional as F

def bce2d(pred, gt, reduction='mean'):
    pos = torch.eq(gt, torch.Tensor([0, 1]).cuda()).float()
    neg= torch.eq(gt, torch.Tensor([1, 0]).cuda()).float()
    num_pos = torch.sum(pos)/2
    num_neg = torch.sum(neg)/2
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    return F.binary_cross_entropy_with_logits(pred, gt, weights, reduction=reduction)