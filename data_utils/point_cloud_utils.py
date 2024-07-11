import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage
from sklearn.decomposition import PCA

def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Perform min/max normalization on points if normalize is True.
    Same as: (x - min(x))/(max(x) - min(x)) for all axes.
    
    Parameters:
    - points: np.ndarray, the point cloud data
    - normalize: bool, whether to normalize the points (default is True)
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

def downsample_voxel_grid(points: np.ndarray, targets: np.ndarray, voxel_size=0.027) -> tuple:
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

def downsample_inverse_planar_aware(points: np.ndarray, targets: np.ndarray, npoints: int = 2000, plane_threshold=40, lower_downsample_rate=0.4, higher_downsample_rate=0.9) -> tuple:
    """
    Increasing plane_threshold will identify larger planar regions 
    Increasing lower_downsample_rate will retain more points in planar regions
    Increasing higher_downsample_rate will retain more points in non-planar regions
    Downsample the point cloud by reducing density in planar regions and preserving density in non-planar regions
    This version accounts for planar regions with thickness using PCA and explained variances
    With NOT normalized data, these work fine: planethreshold = 40     lower downsample 0.4     higher downsample 0.9
    With NORMALIZED data, these work fine: planethreshold = 40     lower downsample 0.4     higher downsample 0.9

    """
    
    # Apply DBSCAN to identify clusters
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
            # Apply PCA to the cluster
            pca = PCA(n_components=3)
            pca.fit(label_points)
            explained_variances = pca.explained_variance_ratio_

            # Identify points close to the principal plane (the first two principal components)
            distances = np.abs(label_points @ pca.components_[2])
            thickness_threshold = np.percentile(distances, 95)  # Adjust this percentile as needed

            planar_mask = distances < thickness_threshold
            planar_points = label_points[planar_mask]
            planar_targets = label_targets[planar_mask]

            non_planar_points = label_points[~planar_mask]
            non_planar_targets = label_targets[~planar_mask]

            # Adjust downsample rates based on explained variances
            planar_downsample_rate = lower_downsample_rate * explained_variances[2]
            non_planar_downsample_rate = higher_downsample_rate * (1 - explained_variances[2])

            # Sample points from planar and non-planar regions separately
            n_planar_samples = max(1, int(len(planar_points) * planar_downsample_rate))
            n_non_planar_samples = max(1, int(len(non_planar_points) * non_planar_downsample_rate))

            planar_choice = np.random.choice(len(planar_points), n_planar_samples, replace=False)
            non_planar_choice = np.random.choice(len(non_planar_points), n_non_planar_samples, replace=False)

            sampled_points.extend(planar_points[planar_choice])
            sampled_targets.extend(planar_targets[planar_choice])

            sampled_points.extend(non_planar_points[non_planar_choice])
            sampled_targets.extend(non_planar_targets[non_planar_choice])
        else:
            # Sample from noise points (label == -1)
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

def estimate_curvature(points, k=20):
    """
    Estimate the curvature of each point in the point cloud.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    curvatures = distances.var(axis=1)
    return curvatures

def downsample_curvature_based(points: np.ndarray, targets: np.ndarray, npoints: int = 15000, curvature_threshold=0.5, min_cluster_size=10, eps=0.5, high_curvature_rate=0.3, low_curvature_rate=0.15) -> tuple:
    """
    Downsample the point cloud by prioritizing points in high-curvature regions (e.g., vehicles).
    """
    
    # Estimate curvature
    curvatures = estimate_curvature(points, k=20)
    
    high_curvature_mask = curvatures > curvature_threshold
    high_curvature_points = points[high_curvature_mask]
    high_curvature_targets = targets[high_curvature_mask]
    
    low_curvature_points = points[~high_curvature_mask]
    low_curvature_targets = targets[~high_curvature_mask]
    
    # Apply DBSCAN to identify clusters in high-curvature points
    clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(high_curvature_points)
    labels = clustering.labels_
    unique_labels = set(labels)

    sampled_points = []
    sampled_targets = []

    for label in unique_labels:
        label_mask = (labels == label)
        label_points = high_curvature_points[label_mask]
        label_targets = high_curvature_targets[label_mask]

        if label != -1:
            # Sample points from clusters with high curvature
            n_samples = max(1, int(len(label_points) * high_curvature_rate))
            choice = np.random.choice(len(label_points), n_samples, replace=False)
            sampled_points.extend(label_points[choice])
            sampled_targets.extend(label_targets[choice])
        else:
            # Sample from noise points (label == -1)
            n_samples = max(1, int(len(label_points) * low_curvature_rate))
            choice = np.random.choice(len(label_points), n_samples, replace=False)
            sampled_points.extend(label_points[choice])
            sampled_targets.extend(label_targets[choice])

    # Sample points from low-curvature regions
    n_low_curvature_samples = max(1, int(len(low_curvature_points) * low_curvature_rate))
    low_curvature_choice = np.random.choice(len(low_curvature_points), n_low_curvature_samples, replace=False)
    sampled_points.extend(low_curvature_points[low_curvature_choice])
    sampled_targets.extend(low_curvature_targets[low_curvature_choice])
    
    sampled_points = np.array(sampled_points)
    sampled_targets = np.array(sampled_targets)

    #if len(sampled_points) > npoints:
    #    choice = np.random.choice(len(sampled_points), npoints, replace=False)
    #    sampled_points = sampled_points[choice]
    #    sampled_targets = sampled_targets[choice]

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

def downsample_biometric(points: np.ndarray, targets: np.ndarray, npoints: int = 2000, edge_threshold=30, downsample_rate=0.6) -> tuple:
    """
    Increase edge_threshold makes the algorithm more selective about what considers edges, so only points with very high gradients will be considered edges.
    Increase downsample_rate allows more non-edge points to be sampled.
    Downsample the point cloud by prioritizing edges and high-gradient areas.
    """

    # Calculate gradients using the Sobel operator
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


def downsample_combined(points: np.ndarray, targets: np.ndarray, npoints : int=7500) -> tuple:
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

def downsample_parallel_combined(points: np.ndarray, targets: np.ndarray, npoints: int = 3000) -> tuple:
    """
    Combine multiple downsampling methods to achieve a balanced downsampling.
    """

    # Apply inverse planar-aware downsampling
    planar_points, planar_targets = downsample_inverse_planar_aware(points, targets, npoints)

    # Apply biometric downsampling
    biometric_points, biometric_targets = downsample_biometric(points, targets, npoints)

    # Apply feature-based downsampling
    feature_points, feature_targets = downsample_feature_based(points, targets, npoints)

    # Combine results from each downsampling method
    combined_points = np.concatenate((planar_points, biometric_points, feature_points), axis=0)
    combined_targets = np.concatenate((planar_targets, biometric_targets, feature_targets), axis=0)

    # Remove duplicate points
    combined_points, unique_indices = np.unique(combined_points, axis=0, return_index=True)
    combined_targets = combined_targets[unique_indices]

    # Ensure the final number of points matches npoints
    if len(combined_points) > npoints:
        choice = np.random.choice(len(combined_points), npoints, replace=False)
        combined_points = combined_points[choice]
        combined_targets = combined_targets[choice]
    elif len(combined_points) < npoints:
        choice = np.random.choice(len(combined_points), npoints, replace=True)
        combined_points = combined_points[choice]
        combined_targets = combined_targets[choice]

    return combined_points, combined_targets
