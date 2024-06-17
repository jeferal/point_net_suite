import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.PointNetV2.point_net_set_abstraction import PointNetSetAbstractionSingleScaleGrouping, PointNetSetAbstractionMultiScaleGrouping
from models.PointNetV2.point_net_feature_propagation import PointNetFeaturePropagation


# ================================================================================================
# ============================    POINTNET ++ CLASSIFICATION FEATURE LEARNER     =================
# ============================            (SINGLE - SCALE GROUPING)              =================
# ================================================================================================
class PointNetV2FeatureLearnerClassificationSingleScaleGrouping(nn.Module):
    ''' PointNet++ Feature Learner for classification module using single - scale grouping '''
    def __init__(self, input_dim=3, extra_feat_dropout=0.0):
        """
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2FeatureLearnerClassificationSingleScaleGrouping, self).__init__()

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
# ============================    POINTNET ++ CLASSIFIER FEATURE LEARNER     =====================
# ============================          (MULTI - SCALE GROUPING)             =====================
# ================================================================================================
class PointNetV2FeatureLearnerClassificationMultiScaleGrouping(nn.Module):
    ''' PointNet++ Feature Learner for classification module using multi - scale grouping '''
    def __init__(self, input_dim=3, extra_feat_dropout=0.0):
        """
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2FeatureLearnerClassificationMultiScaleGrouping, self).__init__()

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
            self.feature_learner = PointNetV2FeatureLearnerClassificationSingleScaleGrouping(input_dim=input_dim, extra_feat_dropout=extra_feat_dropout)
        else:
            self.feature_learner = PointNetV2FeatureLearnerClassificationMultiScaleGrouping(input_dim=input_dim, extra_feat_dropout=extra_feat_dropout)

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
# ============================    POINTNET ++ SEMANTIC SEGMENTATION FEATURE LEARNER     ==========
# ============================                 (SINGLE - SCALE GROUPING)                ==========
# ================================================================================================
class PointNetV2FeatureLearnerSemanticSegSingleScaleGrouping(nn.Module):
    ''' PointNet++ Feature Learner for semantic segmentation module using single - scale grouping '''
    def __init__(self, input_dim=3, extra_feat_dropout=0.0):
        """
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2FeatureLearnerSemanticSegSingleScaleGrouping, self).__init__()

        self.input_dim = input_dim
        self.extra_feat_dropout = extra_feat_dropout

        # 4 layers of PointNet Set Abstractions:
        # From PointNet++ paper: SA(1024, 0.1, [32, 32, 64]) --> SA(256, 0.2, [64, 64, 128]) -->
        #                        SA(64, 0.4, [128, 128, 256]) --> SA(16, 0.8, [256, 256, 512])
        self.sa1 = PointNetSetAbstractionSingleScaleGrouping(npoint=1024, radius=0.1, nsample=32, in_channel=input_dim,
                                                             mlp_layers=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstractionSingleScaleGrouping(npoint=256, radius=0.2, nsample=32, in_channel=64 + 3,
                                                             mlp_layers=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstractionSingleScaleGrouping(npoint=64, radius=0.4, nsample=32, in_channel=128 + 3,
                                                             mlp_layers=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstractionSingleScaleGrouping(npoint=16, radius=0.8, nsample=32, in_channel=256 + 3,
                                                             mlp_layers=[256, 256, 512], group_all=False)

    def forward(self, x, extra_features):
        # Pass through all the PointNet Set Abstractions
        sa1_xyz, sa1_features = self.sa1(x, extra_features)
        sa2_xyz, sa2_features = self.sa2(sa1_xyz, sa1_features)
        sa3_xyz, sa3_features = self.sa3(sa2_xyz, sa2_features)
        sa4_xyz, sa4_features = self.sa4(sa3_xyz, sa3_features)

        return sa1_xyz, sa1_features, sa2_xyz, sa2_features, sa3_xyz, sa3_features, sa4_xyz, sa4_features
    

# ================================================================================================
# ============================    POINTNET ++ SEMANTIC SEGMENTATION FEATURE LEARNER     ==========
# ============================                 (MULTI - SCALE GROUPING)                ==========
# ================================================================================================
class PointNetV2FeatureLearnerSemanticSegMultiScaleGrouping(nn.Module):
    ''' PointNet++ Feature Learner for semantic segmentation module using multi - scale grouping '''
    def __init__(self, input_dim=3, extra_feat_dropout=0.0):
        """
        :param input_dim: number of dimensions of the input, the first 3 ones should be xyz while the rest can be extra features as normals or rgb data
        :param extra_feat_dropout: amount of dropout to apply to the extra features so that the model does not learn only from them
        """
        super(PointNetV2FeatureLearnerSemanticSegMultiScaleGrouping, self).__init__()

        self.input_dim = input_dim
        self.extra_feat_dropout = extra_feat_dropout

        # 4 layers of PointNet Set Abstractions Multi Scale grouping:
        self.sa1 = PointNetSetAbstractionMultiScaleGrouping(npoint=1024, radius=[0.05, 0.1], nsample=[16, 32], in_channel=input_dim,
                                                            mlp_layers=[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMultiScaleGrouping(npoint=256, radius=[0.1, 0.2], nsample=[16, 32], in_channel=64 + 32 + 3,
                                                            mlp_layers=[[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMultiScaleGrouping(npoint=64, radius=[0.2, 0.4], nsample=[16, 32], in_channel=128 + 128 + 3,
                                                            mlp_layers=[[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMultiScaleGrouping(npoint=16, radius=[0.4, 0.8], nsample=[16, 32], in_channel=256 + 256 + 3,
                                                            mlp_layers=[[256, 256, 512], [256, 384, 512]])

    def forward(self, x, extra_features):
        # Pass through all the PointNet Set Abstractions
        sa1_xyz, sa1_features = self.sa1(x, extra_features)
        sa2_xyz, sa2_features = self.sa2(sa1_xyz, sa1_features)
        sa3_xyz, sa3_features = self.sa3(sa2_xyz, sa2_features)
        sa4_xyz, sa4_features = self.sa4(sa3_xyz, sa3_features)

        return sa1_xyz, sa1_features, sa2_xyz, sa2_features, sa3_xyz, sa3_features, sa4_xyz, sa4_features


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

        self.extra_feat_dropout = extra_feat_dropout

        if single_scale_grouping:
            self.feature_learner = PointNetV2FeatureLearnerSemanticSegSingleScaleGrouping(input_dim=input_dim, extra_feat_dropout=extra_feat_dropout)
            input_channels_feature_prop = [768, 384, 320, 128 + (input_dim - 3)] #[512+256, 256+128, 256+64, 128+(input_dim-3)] # Set abstraction output + last layer output
        else:
            self.feature_learner = PointNetV2FeatureLearnerSemanticSegMultiScaleGrouping(input_dim=input_dim, extra_feat_dropout=extra_feat_dropout)
            input_channels_feature_prop = [1536, 512, 352, 128 + (input_dim - 3)] #[512+512+256+256, 128+128+256, 32+64+256, 128+(input_dim-3)] # Set abstraction output + last layer output

        # Feature propagation as the paper states: (last convolution of last feature propagation is outside to activate it with log_softmax instead of relu)
        # FP(256, 256) --> FP(256, 256) --> FP(256, 128) --> FP(128, 128, 128, 128, K)
        self.fp4 = PointNetFeaturePropagation(in_channel=input_channels_feature_prop[0], mlp_layers=[256, 256], dropout_mlp_layers=[0.0, 0.0])
        self.fp3 = PointNetFeaturePropagation(in_channel=input_channels_feature_prop[1], mlp_layers=[256, 256], dropout_mlp_layers=[0.0, 0.0])
        self.fp2 = PointNetFeaturePropagation(in_channel=input_channels_feature_prop[2], mlp_layers=[256, 128], dropout_mlp_layers=[0.0, 0.0])
        self.fp1 = PointNetFeaturePropagation(in_channel=input_channels_feature_prop[3], mlp_layers=[128, 128, 128, 128], dropout_mlp_layers=[0.0, 0.0, 0.0, dropout])
        self.last_conv = nn.Conv1d(128, k, 1)

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

        # Get the features and points of every point abastraction with the feature learner:
        sa1_xyz, sa1_features, sa2_xyz, sa2_features, sa3_xyz, sa3_features, sa4_xyz, sa4_features = self.feature_learner(x, extra_features)

        # Pass through all the feature learners
        fp3_features = self.fp4(sa3_xyz, sa4_xyz, sa3_features, sa4_features)
        fp2_features = self.fp3(sa2_xyz, sa3_xyz, sa2_features, fp3_features)
        fp1_features = self.fp2(sa1_xyz, sa2_xyz, sa1_features, fp2_features)
        fp0_features = self.fp1(x, sa1_xyz, extra_features, fp1_features)

        # Do the last vconvolution to get the classes
        x = self.last_conv(fp0_features)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1)

        return x, sa4_features
    

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
    

class PointNetV2LossForSemanticSegmentation(nn.Module):
    def __init__(self, ce_label_smoothing=0.0, weights=None):
        super(PointNetV2LossForSemanticSegmentation, self).__init__()

        #self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=ce_label_smoothing)
        if weights is None:
            self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=ce_label_smoothing)
        else:
            print("With weights")
            self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, predictions, targets):
        # Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)

        return ce_loss