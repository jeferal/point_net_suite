import torch

import numpy as np

def normalize_points(points : np.ndarray) -> np.ndarray:
        """
            Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
        """
        points = points - points.min(axis=0)
        points /= points.max(axis=0)

        return points

def downsample(points : np.ndarray, targets : np.ndarray, npoints : int = 1024) -> tuple:
    if len(points) > npoints:
        choice = np.random.choice(len(points), npoints, replace=False)
    else:
        # case when there are less points than the desired number
        choice = np.random.choice(len(points), npoints, replace=True)
    points = points[choice, :] 
    targets = targets[choice]

    return points, targets

def get_point_cloud_limits(points : torch.Tensor) -> tuple:
    """
        Get the limits of the point cloud
    """
    min_values = points.min(dim=0)[0]
    max_values = points.max(dim=0)[0]

    return min_values, max_values
