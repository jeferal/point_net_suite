import torch
import torch.nn as nn

from models.point_net import PointNetClassification, PointNetLossForClassification


class get_model(nn.Module):
    def __init__(self, num_points=1024, k=40, dropout=0.4):
        super(get_model, self).__init__()
        self.classificator = PointNetClassification(num_points, k, dropout)

    def forward(self, x):
        return self.classificator(x)

class get_loss(nn.Module):
    def __init__(self, regularization_weight=0.001, gamma=1):
        super(get_loss, self).__init__()
        self.loss_calculator = PointNetLossForClassification(regularization_weight=regularization_weight, gamma=gamma)
    
    def forward(self, pred, target, trans_feat):
        return self.loss_calculator(pred, target, trans_feat)
