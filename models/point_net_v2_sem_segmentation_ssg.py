import torch.nn as nn

from models.PointNetV2.point_net_v2 import PointNetV2SemanticSegmentation, PointNetV2LossForSemanticSegmentation


class get_model(nn.Module):
    def __init__(self, num_points=1024, m=2, dropout=0.5, input_dim=3, extra_feat_dropout=0.0):
        super(get_model, self).__init__()
        self.classificator = PointNetV2SemanticSegmentation(k=m, dropout=dropout, input_dim=input_dim, extra_feat_dropout=extra_feat_dropout,
                                                            single_scale_grouping=True)

    def forward(self, x):
        pred, sa4_features = self.classificator(x)
        return pred, sa4_features, None

class get_loss(nn.Module):
    def __init__(self, label_smoothing=0.0, gamma=1, dice=True, dice_eps=1):
        super(get_loss, self).__init__()
        self.loss_calculator = PointNetV2LossForSemanticSegmentation()
    
    def forward(self, pred, target, pred_choice, loss_weights):
        return self.loss_calculator(pred, target, loss_weights)
