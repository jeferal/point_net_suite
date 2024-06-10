import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PointNetV2.point_net_v2_utils import sample_and_group, sample_and_group_all, index_points, farthest_point_sample, query_ball_point


class PointNetSetAbstractionSingleScaleGrouping(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp_layers, group_all):

        super(PointNetSetAbstractionSingleScaleGrouping, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_dimension = in_channel

        for layer_size in mlp_layers:
            self.mlp_convs.append(nn.Conv2d(last_dimension, layer_size, 1))
            self.mlp_bns.append(nn.BatchNorm2d(layer_size))
            last_dimension = layer_size

        self.group_all = group_all

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points positional data, [B, C, N]
            features: input points features data, [B, D, N]
        Return:
            new_xyz: sampled points positional data, [B, C, S]
            new_features: sample points features data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
            features = features.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_features = sample_and_group_all(xyz, features)
        else:
            new_xyz, new_features = sample_and_group(self.npoint, self.radius, self.nsample, xyz, features)
        # new_xyz: sampled points positional data, [B, npoint, C]
        # new_features: sampled points features data, [B, npoint, nsample, C+D]
        new_features = new_features.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features =  F.relu(bn(conv(new_features)))

        new_features = torch.max(new_features, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_features
    

class PointNetSetAbstractionMultiScaleGrouping(nn.Module):
    def __init__(self, npoint, radii_list, nsample_list, in_channel, mlp_list):

        super(PointNetSetAbstractionMultiScaleGrouping, self).__init__()

        self.npoint = npoint
        self.radii_list = radii_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_dimension = in_channel
            for layer_size in mlp_list[i]:
                convs.append(nn.Conv2d(last_dimension, layer_size, 1))
                bns.append(nn.BatchNorm2d(layer_size))
                last_dimension = layer_size
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points positional data, [B, C, N]
            features: input points features data, [B, D, N]
        Return:
            new_xyz: sampled points positional data, [B, C, S]
            new_features: sample points features data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
            features = features.permute(0, 2, 1)

        B, N, C = xyz.shape
        num_points = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, num_points))
        new_features_list = []
        for i, radius in enumerate(self.radii_list):
            num_sample = self.nsample_list[i]
            group_idx = query_ball_point(radius, num_sample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, num_points, 1, C)
            if features is not None:
                grouped_points = index_points(features, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_features_rad = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_features_list.append(new_features_rad)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features