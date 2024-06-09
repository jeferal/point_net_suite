import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.PointNetV2.point_net_set_abstraction import PointNetSetAbstractionSingleScaleGrouping, PointNetSetAbstractionMultiScaleGrouping


# ================================================================================================
# ============================    POINTNET ++ FEATURE LEARNER     ================================
# ============================     (SINGLE - SCALE GROUPING)      ================================
# ================================================================================================
class PointNetV2FeatureLearnerSingleScaleGrouping(nn.Module):
    ''' PointNet++ Feature Learner module using single - scale grouping '''
    def __init__(self, input_dim=3, extra_feat_dropout=0.0):
        """
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2FeatureLearnerSingleScaleGrouping, self).__init__()

        self.input_dim = input_dim
        self.extra_feat_dropout = extra_feat_dropout

        # 3 layers of PointNet Set Abstractions:
        # From PointNet++ paper: SA(512, 0.2, [64, 64, 128]) --> SA(128, 0.4, [128, 128, 256]) --> SA([256, 512, 1024])
        self.sa1 = PointNetSetAbstractionSingleScaleGrouping(npoint=512, radius=0.2, nsample=32, in_channel=input_dim,
                                                             mlp_layers=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstractionSingleScaleGrouping(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                                             mlp_layers=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstractionSingleScaleGrouping(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                                             mlp_layers=[256, 512, 1024], group_all=True)

    def forward(self, x):
        batchsize, xDimension, _ = x.shape
        #batchsize, dimensions, numpoints = x.shape

        # If we have more than 3 dimensions in the input, the points are only the first 3
        # The rest of the dimensions should be treated as features
        if xDimension > 3:
            extra_features = x[:, 3:, :]
            x = x[:, :3, :]

            # We can also implement some dropout so that the model does not learn to decide based on these extra features
            # (for example, decide a class based on color with rgb extra info). This is done in other models like KPCOnv.
            if self.extra_feat_dropout > 0.0 and np.random.uniform(0, 1) < self.extra_feat_dropout:
                extra_features[:, :, :] = 0.0
        
        else:
            extra_features = None

        # Pass through all the PointNet Set Abstractions
        sa1_xyz, sa1_features = self.sa1(x, extra_features)
        sa2_xyz, sa2_features = self.sa2(sa1_xyz, sa1_features)
        sa3_xyz, sa3_features = self.sa3(sa2_xyz, sa2_features)
        
        x = sa3_features.view(batchsize, 1024)

        return x, sa3_features


# ================================================================================================
# ============================    POINTNET ++ FEATURE LEARNER     ================================
# ============================     (MULTI - SCALE GROUPING)       ================================
# ================================================================================================
class PointNetV2FeatureLearnerMultiScaleGrouping(nn.Module):
    ''' PointNet++ Feature Learner module using single - scale grouping '''
    def __init__(self, input_dim=3, extra_feat_dropout=0.0):
        """
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2FeatureLearnerMultiScaleGrouping, self).__init__()

        self.input_dim = input_dim
        self.extra_feat_dropout = extra_feat_dropout

        # 2 layers of PointNet MultiScaleGrouping Set Abstractions:
        # From PointNet++ paper: SA(512, [0.1, 0.2, 0.4], [[32, 32, 64], [64, 64, 128], [64, 96, 128]]) -->
        #                        SA(128, [0.2, 0.4, 0.8], [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa1 = PointNetSetAbstractionMultiScaleGrouping(npoint=512, radii_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128], in_channel=input_dim,
                                                             mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMultiScaleGrouping(npoint=128, radii_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128], in_channel=320 + 3, #320 = 64 + 128 + 128
                                                             mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # 1 layer of PointNet SingleScaleGrouping Set Abstractions:
        # From PointNet++ paper: SA([256, 512, 1024])
        self.sa3 = PointNetSetAbstractionSingleScaleGrouping(npoint=None, radius=None, nsample=None, in_channel=640 + 3,    #640 = 128 + 256 + 256
                                                             mlp_layers=[256, 512, 1024], group_all=True)

    def forward(self, x):
        batchsize, xDimension, _ = x.shape
        #batchsize, dimensions, numpoints = x.shape

        # If we have more than 3 dimensions in the input, the points are only the first 3
        # The rest of the dimensions should be treated as features
        if xDimension > 3:
            extra_features = x[:, 3:, :]
            x = x[:, :3, :]

            # We can also implement some dropout so that the model does not learn to decide based on these extra features
            # (for example, decide a class based on color with rgb extra info). This is done in other models like KPCOnv.
            if self.extra_feat_dropout > 0.0 and np.random.uniform(0, 1) < self.extra_feat_dropout:
                extra_features[:, :, :] = 0.0
        
        else:
            extra_features = None

        # Pass through all the PointNet Set Abstractions
        sa1_xyz, sa1_features = self.sa1(x, extra_features)
        sa2_xyz, sa2_features = self.sa2(sa1_xyz, sa1_features)
        sa3_xyz, sa3_features = self.sa3(sa2_xyz, sa2_features)
        
        x = sa3_features.view(batchsize, 1024)

        return x, sa3_features
    

# ================================================================================================
# ============================    POINTNET ++ CLASSIFICATION     =================================
# ================================================================================================
class PointNetV2Classification(nn.Module):
    ''' PointNet++ Classification module that obtains the object class '''
    def __init__(self, k=2, dropout=0.5, input_dim=3, extra_feat_dropout=0.0, single_scale_grouping=False):
        """
        :param num_points: number of points in the point cloud
        :param k: number of object classes available
        :param dropout: dropout amount to apply after every fc layer
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2Classification, self).__init__()

        if single_scale_grouping:
            self.feature_learner = PointNetV2FeatureLearnerSingleScaleGrouping(input_dim=input_dim, extra_feat_dropout=extra_feat_dropout)
        else:
            self.feature_learner = PointNetV2FeatureLearnerMultiScaleGrouping(input_dim=input_dim, extra_feat_dropout=extra_feat_dropout)

        # First FC layer:
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)

        # Second FC layer:
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)

        # Last FC layer:
        self.fc3 = nn.Linear(256, k)

    def forward(self, x):
        # Get the features extracted with the feature extractor:
        x, sa3_features = self.feature_learner(x)

        # Pass through the three FC layers
        x = F.relu(self.bn1(self.dropout1(self.fc1(x))))
        x = F.relu(self.bn2(self.dropout2(self.fc2(x))))
        x = self.fc3(x)
        output = F.log_softmax(x, -1)
        
        return output, sa3_features


# ================================================================================================
# ============================    POINTNET ++ SEMANTIC SEGMENTATION     ==========================
# ================================================================================================
class PointNetV2SemanticSegmentation(nn.Module):
    ''' PointNet++ Semantic Segmentation module that obtains every point class in a scene '''
    def __init__(self, k=2, dropout=0.5, input_dim=3, extra_feat_dropout=0.0, single_scale_grouping=False):
        """
        :param num_points: number of points in the point cloud
        :param k: number of object classes available
        :param dropout: dropout amount to apply after the first shared mlp
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2SemanticSegmentation, self).__init__()


    def forward(self, x):
        return
    

# ================================================================================================
# ============================    POINTNET ++ LOSS     ===========================================
# ================================================================================================
class PointNetV2LossForClassification(nn.Module):
    def __init__(self, ce_label_smoothing=0.0):
        super(PointNetV2LossForClassification, self).__init__()

        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=ce_label_smoothing)

    def forward(self, predictions, targets):
        # Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions, targets)

        return ce_loss