import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage

def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Perform min/max normalization on points.
    Same as: (x - min(x))/(max(x) - min(x))
    """
    points = points - points.min(axis=0)
    points /= points.max(axis=0)
    return points

def downsample(points: np.ndarray, targets: np.ndarray, npoints: int = 1024) -> tuple:
    """
    Randomly downsample the point cloud to a specified number of points.
    """
    if len(points) > npoints:
        choice = np.random.choice(len(points), npoints, replace=False)
    else:
        choice = np.random.choice(len(points), npoints, replace=True)
    points = points[choice, :]
    targets = targets[choice]
    return points, targets

def get_point_cloud_limits(points: torch.Tensor) -> tuple:
    """
    Get the limits of the point cloud.
    """
    min_values = points.min(dim=0)[0]
    max_values = points.max(dim=0)[0]
    return min_values, max_values

def downsample_voxel_grid(points: np.ndarray, targets: np.ndarray, voxel_size=0.01) -> tuple:
    """
    Downsample the point cloud using a voxel grid approach.
    """
    min_coords = points.min(axis=0)
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
    voxel_dict = {}

    # Create dictionary to gather points and targets in each voxel
    for idx, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(idx)

    sampled_points = []
    sampled_targets = []

    # Aggregate data for each voxel
    for voxel_key, indices in voxel_dict.items():
        if len(indices) == 0:
            continue

        voxel_points = points[indices]
        voxel_targets = targets[indices].astype(int)  # Ensure targets are integers

        # Check if all voxel points have the same number of dimensions
        if not all(len(point) == len(voxel_points[0]) for point in voxel_points):
            print(f"Inconsistent point dimensions in voxel {voxel_key}: {voxel_points}")
            continue

        # Compute centroid and majority target
        centroid = np.mean(voxel_points, axis=0)
        majority_target = np.bincount(voxel_targets).argmax()

        sampled_points.append(centroid)
        sampled_targets.append(majority_target)

    # Convert lists of centroids and majority targets to numpy arrays
    sampled_points = np.array(sampled_points)
    sampled_targets = np.array(sampled_targets)
    return sampled_points, sampled_targets


def downsample_inverse_planar_aware(points: np.ndarray, targets: np.ndarray, npoints: int = 8000, plane_threshold=0.9, lower_downsample_rate=0.8, higher_downsample_rate=0.2) -> tuple:
    """
    Downsample the point cloud by reducing density in planar regions and preserving density in non-planar regions.
    """
    clustering = DBSCAN(eps=plane_threshold, min_samples=10).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)

    sampled_points = []
    sampled_targets = []

    for label in unique_labels:
        label_mask = (labels == label)
        label_points = points[label_mask]
        label_targets = targets[label_mask]

        if label != -1:
            n_samples = max(1, int(len(label_points) * lower_downsample_rate))
        else:
            n_samples = max(1, int(len(label_points) * higher_downsample_rate))

        choice = np.random.choice(len(label_points), n_samples, replace=False)
        sampled_points.extend(label_points[choice])
        sampled_targets.extend(label_targets[choice])

    sampled_points = np.array(sampled_points)
    sampled_targets = np.array(sampled_targets)

    if len(sampled_points) > npoints:
        choice = np.random.choice(len(sampled_points), npoints, replace=False)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]
    return sampled_points, sampled_targets

def downsample_feature_based(points: np.ndarray, targets: np.ndarray, npoints: int = 8000, k=20, high_variance_threshold=0.05, low_variance_rate=0.5) -> tuple:
    """
    Downsample the point cloud by prioritizing points in high-variance regions.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    variance = distances.var(axis=1)

    high_variance_mask = variance > high_variance_threshold
    high_variance_points = points[high_variance_mask]
    high_variance_targets = targets[high_variance_mask]

    low_variance_points = points[~high_variance_mask]
    low_variance_targets = targets[~high_variance_mask]

    n_low_variance_samples = int(len(low_variance_points) * low_variance_rate)
    low_variance_choice = np.random.choice(len(low_variance_points), n_low_variance_samples, replace=False)

    sampled_points = np.concatenate((high_variance_points, low_variance_points[low_variance_choice]))
    sampled_targets = np.concatenate((high_variance_targets, low_variance_targets[low_variance_choice]))
    if len(sampled_points) > npoints:
        choice = np.random.choice(len(sampled_points), npoints, replace=False)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]
    elif len(sampled_points) < npoints:
        choice = np.random.choice(len(sampled_points), npoints, replace=True)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]

    return sampled_points, sampled_targets

def downsample_biometric(points: np.ndarray, targets: np.ndarray, npoints: int = 8000, edge_threshold=0.01, downsample_rate=0.5) -> tuple:
    """
    Downsample the point cloud by prioritizing edges and high-gradient areas.
    """
    gradients = np.sqrt(np.sum(np.square(scipy.ndimage.sobel(points, axis=0)), axis=1))
    edge_mask = gradients > edge_threshold
    edge_points = points[edge_mask]
    edge_targets = targets[edge_mask]

    non_edge_points = points[~edge_mask]
    non_edge_targets = targets[~edge_mask]

    n_non_edge_samples = int(len(non_edge_points) * downsample_rate)
    non_edge_choice = np.random.choice(len(non_edge_points), n_non_edge_samples, replace=False)

    sampled_points = np.concatenate((edge_points, non_edge_points[non_edge_choice]))
    sampled_targets = np.concatenate((edge_targets, non_edge_targets[non_edge_choice]))

    if len(sampled_points) > npoints:
        choice = np.random.choice(len(sampled_points), npoints, replace=False)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]
    elif len(sampled_points) < npoints:
        choice = np.random.choice(len(sampled_points), npoints, replace=True)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]

    return sampled_points, sampled_targets

def downsample_combined(points: np.ndarray, targets: np.ndarray, npoints : int=2000) -> tuple:
    """
    Combine multiple downsampling methods to achieve a balanced downsampling.
    """
    # Apply voxel grid downsampling
    points, targets = downsample_voxel_grid(points, targets)
    
    # Apply inverse planar-aware downsampling
    points, targets = downsample_inverse_planar_aware(points, targets, npoints)
    
    # Apply biometric downsampling if needed
    if len(points) > npoints:
        points, targets = downsample_biometric(points, targets, npoints)
    
    # Apply feature-based downsampling if needed
    if len(points) > npoints:
        points, targets = downsample_feature_based(points, targets, npoints)
    
    return points, targets


def downsample_test(points: np.ndarray, targets: np.ndarray, npoints : int = 2000, voxel_size=0.1, plane_threshold=0.02, higher_downsample_rate=0.5) -> tuple:
    """
    Test downsampling combining voxel grid and planar aware methods.
    """
    points, targets = downsample_voxel_grid(points, targets, voxel_size)
    points, targets = downsample_inverse_planar_aware(points, targets, npoints, plane_threshold, higher_downsample_rate)
    return points, targets
