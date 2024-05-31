import torch.nn as nn
from models.point_net import PointNetSegmentation, PointNetLossForSemanticSegmentation


class get_model(nn.Module):
    def __init__(self, num_points=1024, m=2, dropout=0.4, input_dim=3):
        super(get_model, self).__init__()
        self.classificator = PointNetSegmentation(num_points, m, dropout=dropout, input_dim=input_dim) #Maybe add dropout to the last layer?

    def forward(self, x):
        return self.classificator(x)

class get_loss(nn.Module):
    def __init__(self, label_smoothing=0.0, gamma=1, dice=True, dice_eps=1):
        super(get_loss, self).__init__()
        self.loss_calculator = PointNetLossForSemanticSegmentation(ce_label_smoothing=label_smoothing, gamma=gamma, dice=dice, dice_eps=dice_eps)
    
    def forward(self, pred, target, pred_choice):
        return self.loss_calculator(pred, target, pred_choice)
