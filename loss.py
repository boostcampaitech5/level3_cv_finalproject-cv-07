import torch
import os
import random
import numpy as np


# my version of triplet loss
def triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = ((anchor-positive)**2).sum(1).sqrt() # we need to create tensor to be "[value]" rather than just "value"
    negative_distance = ((anchor-negative)**2).sum(1).sqrt()
    loss = torch.relu(margin + positive_distance - negative_distance)
    return loss.mean()


class QuadrupletLoss(torch.nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples. The negative examples needs not to be matching the anchor, the positive and each other.
    """
    def __init__(self, margin1=2.0, margin2=1.0):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):

        squarred_distance_pos = (anchor - positive).pow(2).sum(1)
        squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
        squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)

        quadruplet_loss = F.relu(self.margin1 + squarred_distance_pos - squarred_distance_neg) + F.relu(self.margin2 + squarred_distance_pos - squarred_distance_neg_b)

        return quadruplet_loss.mean()