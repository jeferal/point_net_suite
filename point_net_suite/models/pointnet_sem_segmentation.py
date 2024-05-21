import torch.nn as nn
from point_net_suite.models.point_net import PointNetSegmentation, PointNetLossForSemanticSegmentation


class get_model(nn.Module):
    def __init__(self, num_points=1024, m=2):
        super(get_model, self).__init__()
        self.classificator = PointNetSegmentation(num_points, m) #Maybe add dropout to the last layer?

    def forward(self, x):
        return self.classificator(x)

class get_loss(nn.Module):
    def __init__(self, gamma=1, dice=True, dice_eps=1):
        super(get_loss, self).__init__()
        self.loss_calculator = PointNetLossForSemanticSegmentation(gamma=gamma, dice=dice, dice_eps=dice_eps)
    
    def forward(self, pred, target, pred_choice):
        return self.loss_calculator(pred, target, pred_choice)
