import torch.nn as nn

from models.PointNet.point_net import PointNetSemanticSegmentation, PointNetLossForSemanticSegmentation


class get_model(nn.Module):
    def __init__(self, num_points=1024, m=2, dropout=0.4, input_dim=3, extra_feat_dropout=0.0):
        super(get_model, self).__init__()
        self.classificator = PointNetSemanticSegmentation(num_points, m, dropout=dropout, input_dim=input_dim, extra_feat_dropout=extra_feat_dropout) #Maybe add dropout to the last layer?

    def forward(self, x):
        return self.classificator(x)

class get_loss(nn.Module):
    def __init__(self, label_smoothing=0.0, gamma=1, dice=True, dice_eps=1):
        super(get_loss, self).__init__()
        self.loss_calculator = PointNetLossForSemanticSegmentation(ce_label_smoothing=label_smoothing, gamma=gamma, dice=dice, dice_eps=dice_eps)
    
    def forward(self, pred, target, pred_choice, loss_weights):
        return self.loss_calculator(pred, target, pred_choice)
