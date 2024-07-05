

def downsample_planar_aware(self, points, targets, plane_threshold=0.02, higher_downsample_rate=0.5):
    from sklearn.cluster import DBSCAN

    # DBSCAN clustering to identify planar regions
    clustering = DBSCAN(eps=plane_threshold, min_samples=10).fit(points)
    # labels are assigned to each point, where -1 indicates noise
    labels = clustering.labels_

    unique_labels = set(labels)  # unique cluster labels
    sampled_points = []
    sampled_targets = []

    for label in unique_labels:
        label_mask = (labels == label)  # mask for current cluster
        label_points = points[label_mask]
        label_targets = targets[label_mask]
        
        if label != -1:  # For planar surfaces (non-noise)
            # Downsampling rate for planar surfaces
            # n_samples is the number of points to retain
            n_samples = max(1, int(len(label_points) * higher_downsample_rate))
            # Randomly select n_samples points from the cluster
            choice = np.random.choice(len(label_points), n_samples, replace=False)
        else:
            # Lower downsampling rate for non-planar regions (complex features)
            # Here, we use a fixed number of points per unique label cluster
            choice = np.random.choice(len(label_points), min(len(label_points), self.npoints // len(unique_labels)), replace=False)

        sampled_points.extend(label_points[choice])  # add selected points
        sampled_targets.extend(label_targets[choice])  # add corresponding targets
    
    sampled_points = np.array(sampled_points)
    sampled_targets = np.array(sampled_targets)

    # Ensure the number of points matches npoints
    if len(sampled_points) > self.npoints:
        choice = np.random.choice(len(sampled_points), self.npoints, replace=False)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]
    elif len(sampled_points) < self.npoints:
        choice = np.random.choice(len(sampled_points), self.npoints, replace=True)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]

    return sampled_points, sampled_targets




def downsample_feature_based(self, points, targets, k=20, high_variance_threshold=0.05, low_variance_rate=0.5):
    from sklearn.neighbors import NearestNeighbors

    # Compute local density variance
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    variance = distances.var(axis=1)  # variance of distances to k nearest neighbors

    # Separate points into high and low variance regions
    high_variance_mask = variance > high_variance_threshold
    high_variance_points = points[high_variance_mask]
    high_variance_targets = targets[high_variance_mask]
    low_variance_points = points[~high_variance_mask]
    low_variance_targets = targets[~high_variance_mask]

    # Downsample low variance regions more aggressively
    n_low_variance_samples = int(len(low_variance_points) * low_variance_rate)
    # Randomly select n_low_variance_samples points from low variance regions
    low_variance_choice = np.random.choice(len(low_variance_points), n_low_variance_samples, replace=False)
    
    # Combine high and low variance samples
    sampled_points = np.concatenate((high_variance_points, low_variance_points[low_variance_choice]))
    sampled_targets = np.concatenate((high_variance_targets, low_variance_targets[low_variance_choice]))

    # Ensure the number of points matches npoints
    if len(sampled_points) > self.npoints:
        choice = np.random.choice(len(sampled_points), self.npoints, replace=False)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]
    elif len(sampled_points) < self.npoints:
        choice = np.random.choice(len(sampled_points), self.npoints, replace=True)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]

    return sampled_points, sampled_targets


def downsample_biometric(self, points, targets, edge_threshold=0.01, downsample_rate=0.5):
    import scipy.ndimage

    # Compute the gradient (magnitude) of the point cloud
    gradients = np.sqrt(np.sum(np.square(scipy.ndimage.sobel(points, axis=0)), axis=1))
    
    # Identify edge points based on gradient magnitude
    edge_mask = gradients > edge_threshold
    
    # Separate edge and non-edge points
    edge_points = points[edge_mask]
    edge_targets = targets[edge_mask]
    non_edge_points = points[~edge_mask]
    non_edge_targets = targets[~edge_mask]
    
    # Downsample non-edge points more aggressively
    n_non_edge_samples = int(len(non_edge_points) * downsample_rate)
    non_edge_choice = np.random.choice(len(non_edge_points), n_non_edge_samples, replace=False)
    
    # Combine edge and downsampled non-edge points
    sampled_points = np.concatenate((edge_points, non_edge_points[non_edge_choice]))
    sampled_targets = np.concatenate((edge_targets, non_edge_targets[non_edge_choice]))
    
    # Ensure the number of points matches npoints
    if len(sampled_points) > self.npoints:
        choice = np.random.choice(len(sampled_points), self.npoints, replace=False)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]
    elif len(sampled_points) < self.npoints:
        choice = np.random.choice(len(sampled_points), self.npoints, replace=True)
        sampled_points = sampled_points[choice]
        sampled_targets = sampled_targets[choice]

    return sampled_points, sampled_targets


#https://ar5iv.labs.arxiv.org/html/2302.14673
#https://www.mdpi.com/2076-3417/14/8/3160

def downsample_combined(self, points, targets, final_npoints=2000):
    # Step 1: Voxel Grid Downsampling
    ###
    # To effectively downsample our pointcloud of 200000 points (depending of the tiles "ask Jesús")
    # we can combine the three algorithms—Voxel Grid Downsampling, Biometric-Inspired Downsampling,
    # and Feature-Based Adaptive Downsampling—in series. This approahc aims to get the best strengths
    # of each algorithm to progressively reduce the point cloud size while preserving important features
    #
    # Step-by-Step Approach:
    #
    # Initial Reduction with Voxel Grid Downsampling:
    # - This method quickly reduces the point cloud size by dividing the space into a voxel grid
    #   I thought to use this as a replacement of the uniform downsampling method used by default previously
    #   By selecting a representative point from each occupied voxel. It ensures that the spatial
    #   distribution of the points is preserved and provides a good initial reduction, or so we should expect lol
    # - Expected Output: A reduced point cloud with approximately 50,000 points, we can adapt this
    ###
    points, targets = self.downsample_voxel_grid(points, targets)

    # Step 2: Biometric-Inspired Downsampling
    # - We could aslo apply the the Biometric-Inspired Downsampling to the reduced previous point cloud. This method
    #   will focus on retaining points that are critical for recognizing edges and contours, sort of like how we had the critical points
    #   with the ModelNet40, tries to replicate that algorithm behaviour to help understand scene more important features... ideally
    # - Expected Output: A further reduced point cloud with roughly 10,000 points,
    #   retaining key features and edges.
    points, targets = self.downsample_biometric(points, targets)

    # Step 3: Feature-Based Adaptive Downsampling
    #   We will downsample it even more with to further refine the point cloud
    #   This method uses local density variance to adaptively downsample, preserving points in
    #   regions with high variance (complex features) and reducing points in uniform areas
    # - Expected Output: A final point cloud with the desired number of points, e.g., 2,000 points,
    #   with a good balance of preserved features and reduced data size

    ####Note: I have another idea but it's harder to implement, lets first try this and see how it evolves after a few tests/runs...
    points, targets = self.downsample_feature_based(points, targets)

    # Final adjustment to match the final number of points
    if len(points) > final_npoints:
        choice = np.random.choice(len(points), final_npoints, replace=False)
        points = points[choice]
        targets = targets[choice]
    elif len(points) < final_npoints:
        choice = np.random.choice(len(points), final_npoints, replace=True)
        points = points[choice]
        targets = targets[choice]

    return points, targets
