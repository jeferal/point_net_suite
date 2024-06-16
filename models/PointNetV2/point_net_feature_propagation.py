import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PointNetV2.point_net_v2_utils import square_distance, index_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp_layers, dropout_mlp_layers):

        super(PointNetFeaturePropagation, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_drops = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_dimension = in_channel
        for i, layer_size in enumerate(mlp_layers):
            self.mlp_convs.append(nn.Conv1d(last_dimension, layer_size, 1))
            self.mlp_drops.append(nn.Dropout(dropout_mlp_layers[i]))
            self.mlp_bns.append(nn.BatchNorm1d(layer_size))
            last_dimension = layer_size

    def forward(self, xyz1, xyz2, features1, features2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            features1: input points features data, [B, D, N]
            features2: sampled input points features data, [B, D, S]
        Return:
            new_xyz: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        features2 = features2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = features2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(features2, idx) * weight.view(B, N, 3, 1), dim=2)

        if features1 is not None:
            features1 = features1.permute(0, 2, 1)
            new_xyz = torch.cat([features1, interpolated_points], dim=-1)
        else:
            new_xyz = interpolated_points

        new_xyz = new_xyz.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            drop = self.mlp_drops[i]
            bn = self.mlp_bns[i]
            new_xyz = F.relu(bn(drop(conv(new_xyz))))
        return new_xyz