import torch
import torch.nn as nn

from models.PointNetV2.point_net_v2 import PointNetV2Classification, PointNetV2LossForClassification


class get_model(nn.Module):
    def __init__(self, num_points=1024, k=40, dropout=0.4, input_dim=3, extra_feat_dropout=0.0, useDensityFps=False):
        super(get_model, self).__init__()
        self.classificator = PointNetV2Classification(k=k, dropout=dropout, input_dim=input_dim, extra_feat_dropout=extra_feat_dropout,
                                                      single_scale_grouping=False, useDensityFps=useDensityFps)

    def forward(self, x):
        pred, sa3_features = self.classificator(x)
        return pred, sa3_features, None

class get_loss(nn.Module):
    def __init__(self, label_smoothing=0.0, regularization_weight=0.001, gamma=1):
        super(get_loss, self).__init__()
        self.loss_calculator = PointNetV2LossForClassification(ce_label_smoothing=label_smoothing)
    
    def forward(self, pred, target, trans_feat):
        return self.loss_calculator(pred, target)
