import torch
import torch.nn as nn

from models.PointNet.point_net import PointNetClassification, PointNetLossForClassification


class get_model(nn.Module):
    def __init__(self, num_points=1024, k=40, dropout=0.4, input_dim=3, extra_feat_dropout=0.0):
        super(get_model, self).__init__()
        self.classificator = PointNetClassification(num_points, k, dropout, input_dim=input_dim, extra_feat_dropout=extra_feat_dropout)

    def forward(self, x):
        return self.classificator(x)

class get_loss(nn.Module):
    def __init__(self, label_smoothing=0.0, regularization_weight=0.001, gamma=1):
        super(get_loss, self).__init__()
        self.loss_calculator = PointNetLossForClassification(ce_label_smoothing=label_smoothing, regularization_weight=regularization_weight, gamma=gamma)
    
    def forward(self, pred, target, trans_feat):
        return self.loss_calculator(pred, target, trans_feat)
