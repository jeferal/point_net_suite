import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================================
# ==================================    T-NET    =================================================
# ================================================================================================
class Tnet(nn.Module):
    ''' T-Net module that learns a Transformation matrix with any specified dimension '''
    def __init__(self, dim, num_points=1024):
        """
        :param dim: the dimension of input features
        :param num_points: number of points in the point cloud
        """
        super(Tnet, self).__init__()

        # Dimensions
        self.dim = dim 

        # Shared MLPs are implemented as 1D convolutions to make implementation easier (as convolutions already share weights)
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        # Batch Normalization required for the shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Max pool between the shared MLPs and the FC layers
        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

        # Fully conected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim*dim)

        # Batch Normalization for the FC layers
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.shape[0]

        # Pass through shared MLPs:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Do the max pool
        x = self.max_pool(x)

        # Flatten x to be able to be passed to the first FC layer
        x = x.view(batchsize, -1)
        
        # Pass through FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Reshape the output of the last FC layer (to be dim x dim again)
        x = x.view(-1, self.dim, self.dim)

        # Create an identity matrix with the current dimension
        iden = torch.eye(self.dim, requires_grad=True).repeat(batchsize, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        # Add the identity matrix to the output
        x = x + iden

        return x