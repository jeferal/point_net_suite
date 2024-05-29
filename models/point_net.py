import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================================
# ==================================    POINTNET ENCODER    ======================================
# ================================================================================================
class PointNetEncoder(nn.Module):
    ''' PointNet Encoder module that obtains the global and local point features '''
    def __init__(self, num_points=1024, local_feat=False ):
        """
        :param num_points: number of points in the point cloud
        :param local_feat: if True, forward() returns the concatenation of the local and global features
        """
        super(PointNetEncoder, self).__init__()

        self.num_points = num_points
        self.local_feat = local_feat

        # First Spatial Transformer Network (T-net) and shared MLP (with 2 layers (64,64))
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # Batch Normalization required for the first shared MLP
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Second Spatial Transformer Network (T-net) and shared MLP (with 3 layers (64,128,1024))
        self.tnet2 = Tnet(dim=64, num_points=num_points)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        
        # Batch Normalization required for the second shared MLP
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Final max pool to get the global features (returning indices lets us visualize the critical points later!)
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)
        # This might cause warnigns, we can them with: import warnings warnings.filterwarnings("ignore")

    def forward(self, x):
            batchsize = x.shape[0]
            xDimension = x.shape[1]
            #batchsize, dimensions, numpoints = x.size()

            # If we have more than 3 dimensions in the input, the points are only the first 3
            # The rest of the dimensions should be treated as features and we cannot transform them like we do with points
            if xDimension > 3:
                extra_features = x[:, :, 3:]
                x = x[:, :, :3]
            
            # Pass through first Tnet to get the input transform matrix
            input_transform_matrix = self.tnet1(x)

            # Perform the first transformation across every point (3 dims)
            x = torch.bmm(x.transpose(2, 1), input_transform_matrix).transpose(2, 1)

            # If we had more than 3 dimensions, after doing the points tranformation, we can add back the extra dimension and they will be treated as features
            if xDimension > 3:
                x = torch.cat([x, extra_features], dim=2)

            # Pass through the first shared MLP
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            
            # Pass through second Tnet, getting the feature transform matrix
            feature_transform_matrix = self.tnet2(x)

            # Perform the second transformation across each feature (in 64 dims now) 
            x = torch.bmm(x.transpose(2, 1), feature_transform_matrix).transpose(2, 1)

            # Save the local point features (used in the segmentation head)
            if self.local_feat:
                local_features = x.clone()

            # Pass through the second shared MLP
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))

            # Get the global feature and the critical points indexes using the max pool, reshape them into a vector
            global_features, critical_indexes = self.max_pool(x)
            global_features = global_features.view(batchsize, -1)
            critical_indexes = critical_indexes.view(batchsize, -1)

            # Return the concatenation of local and global features if needed. Return the global features only otherwise
            if self.local_feat:
                combined_features = torch.cat((local_features, 
                                                global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), 
                                                dim=1)

                return combined_features, critical_indexes, feature_transform_matrix

            else:
                return global_features, critical_indexes, feature_transform_matrix


# ================================================================================================
# ==================================    POINTNET CLASSIFICATION    ===============================
# ================================================================================================
class PointNetClassification(nn.Module):
    ''' PointNet Classification module that obtains the object class using the global features '''
    def __init__(self, num_points=1024, k=2, dropout=0.4):
        """
        :param num_points: number of points in the point cloud
        :param k: number of object classes available
        :param dropout: dropout amount to apply after the second fc layer
        """
        super(PointNetClassification, self).__init__()

        # PointNet Encoder with only global features
        self.encoder = PointNetEncoder(num_points, local_feat=False)

        # FC layers for the classification MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        # Batch Normalization for the first 2 fc layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # In the paper only batch norm was added to the second fc layer (before the classication layer), 
        # but other versions add dropout to that second fc layer
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, x):
        # Get the global features from the encoder (also the critical points and feature transform matrix)
        global_features, critical_indexes, feature_transform_matrix = self.encoder(x) 

        # Pass through the MLP
        x = F.relu(self.bn1(self.fc1(global_features)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output_scores = self.fc3(x)

        # Return the output scores per class, critical points and feature transform matrix
        return output_scores, critical_indexes, feature_transform_matrix


# ================================================================================================
# ==================================    POINTNET SEGMENTATION    =================================
# ================================================================================================
class PointNetSegmentation(nn.Module):
    ''' PointNet Segmentation module that obtains the object part subclass using the global and local features combined '''
    def __init__(self, num_points=1024, m=2):
        """
        :param num_points: number of points in the point cloud
        :param k: number of object part classes available
        """
        super(PointNetSegmentation, self).__init__()

        # PointNet Encoder with only global features
        self.encoder = PointNetEncoder(num_points, local_feat=True)

        # FC layers for the first shared MLP
        self.conv1 = nn.Conv1d(1088, 512, kernel_size=1) #1088 comes from concatenating 64 local features vector with the 1024 global feature vector
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)

        # Batch Normalization for the first shared MLP fc layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # Last shared MLP that obtais the part classes
        self.conv4 = nn.Conv1d(128, m, kernel_size=1)
        

    def forward(self, x):
        # Get the local + global features from the encoder (also the critical points and feature transform matrix)
        combined_features, critical_indexes, feature_transform_matrix = self.encoder(x) 

        # Pass through the first shared MLP
        x = F.relu(self.bn1(self.conv1(combined_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        point_features = F.relu(self.bn3(self.conv3(x)))

        # Pass through the last shared MLP
        output_scores = self.conv4(point_features)

        output_scores = output_scores.transpose(2, 1)

        # Return the output scores per class matrix (probabilities of every class), critical points and feature transform matrix
        return output_scores, critical_indexes, feature_transform_matrix
    

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
    

# ================================================================================================
# ==================================    POINTNET LOSS    =========================================
# ================================================================================================
class PointNetLossForClassification(nn.Module):
    def __init__(self, regularization_weight=0.001, gamma = 1):
        super(PointNetLossForClassification, self).__init__()

        self.regularization_weight = regularization_weight
        self.gamma = gamma

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, feat_transforms=None):
        batchsize = predictions.shape[0]

        # Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions, targets)

        # Get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # get regularization term
        if self.regularization_weight > 0 and feat_transforms is not None:
            I = torch.eye(64).unsqueeze(0).repeat(feat_transforms.shape[0], 1, 1)
            if feat_transforms.is_cuda: I = I.cuda()
            reg = torch.linalg.norm(I - torch.bmm(feat_transforms, feat_transforms.transpose(2, 1)))
            reg = self.regularization_weight*reg/batchsize
        else:
            reg = 0

        # Compute loss
        loss = ((1 - pn)**self.gamma * ce_loss)
        
        return loss.mean() + reg
    

class PointNetLossForSemanticSegmentation(nn.Module):
    def __init__(self, gamma = 1, dice=True, dice_eps=1):
        super(PointNetLossForSemanticSegmentation, self).__init__()

        self.gamma = gamma
        self.dice = dice
        self.dice_eps = dice_eps

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, pred_choice):
        batchsize = predictions.shape[0]

        # Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)

        # Reformat predictions (b, n, c) -> (b*n, c)
        predictions = predictions.contiguous() \
                        .view(-1, predictions.size(2))
        
        # Get predicted class probabilities for the true class
        pn = F.softmax(predictions, dim=1)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # Compute loss
        loss = ((1 - pn)**self.gamma * ce_loss)
        
        loss = loss.mean()

        if self.dice:
            # Compute dice loss
            pred_choice_flat = pred_choice.reshape(-1)
            targets_flat = targets.reshape(-1)

            classes = torch.unique(targets_flat)

            top = 0
            bot = 0
            for c in classes:
                locs = targets_flat == c

                # Get truth and predictions for each class
                y_true = targets_flat[locs]
                y_hat = pred_choice_flat[locs]

                top += torch.sum(y_hat == y_true)
                bot += len(y_true) + len(y_hat)

            dice_loss = 1 - 2*((top + self.dice_eps)/(bot + self.dice_eps))
            return loss + dice_loss
        
        else:
            return loss
