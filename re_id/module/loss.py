import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = ((anchor-positive)**2).sum(1)
    negative_distance = ((anchor-negative)**2).sum(1)
    loss = torch.relu(margin + positive_distance - negative_distance)
    return loss.mean()

def quadruplet_loss(anchor, positive, negative1, negative2, margin1=2.0, margin2=1.0):
    squarred_distance_pos = (anchor - positive).pow(2).sum(1)
    squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
    squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)
    quadruplet_loss = F.relu(margin1 + squarred_distance_pos - squarred_distance_neg) + F.relu(margin2 + squarred_distance_pos - squarred_distance_neg_b)
    return quadruplet_loss.mean()