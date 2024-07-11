import os

from plyfile import PlyData

import json

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from multiprocessing import Lock
from functools import partial

import open3d as o3d

import torch
from torch.utils.data import Dataset

from sklearn.utils import compute_class_weight

from data_utils.point_cloud_utils import *
from data_utils.metrics import compute_class_distribution


def init_locks(l):
    # Define as many locks as the number of tiles
    # And store them in a dictionary
    global lock_map
    lock_map = l

class DalesDataset(Dataset):

    CATEGORIES = [
        'ground',
        'vegetation',
        'car',
        'truck',
        'powerline',
        'fence',
        'pole',
        'buildings'
    ]

    def __init__(self, root : str, split : str, partitions = 1, intensity : bool = False, instance_seg : bool = False, overlap : float = 0.0, npoints : int = 20000, normalize: bool = True, **kwargs):
        self._root = root
        self._split = split

        self._npoints = npoints
        self._normalize = normalize

        # Get the beta parameter
        beta = kwargs.get('beta', 0.999)
        weight_type = kwargs.get('weight_type', None)
        self._downsampling_method = kwargs.get('downsampling_method', 'uniform')

        # Create the data directory
        self._data_dir = os.path.join(self._root, self._split)
        self._cache_dir = os.path.join(self._root, self._split, "cache")
        self._overlap = round(overlap, 2)
        if overlap < 0.0 or overlap > 1.0:
            raise ValueError("Overlap must be in the range [0, 1]")

        # Create the cache directory if it does not exist
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        # Create the list of files
        self._ply_files = []
        for file in os.listdir(self._data_dir):
            if file.startswith("."):
                continue
            if file.endswith(".ply"):
                self._ply_files.append(file)

        self._intensity = intensity
        self._instance_seg = instance_seg

        if self._instance_seg:
            raise NotImplementedError("Instance segmentation is not implemented yet")

        self._partitions = partitions

        # List with the path of the txt files
        self._split_files = []
        # Loop for every file
        print(f"Splitting the point cloud into {self._partitions} x {self._partitions} tiles...")
        for ply_file in tqdm(self._ply_files):

            # Cache path
            cache_dir = os.path.join(self._cache_dir,
                                     f"{os.path.splitext(ply_file)[0]}_{self._partitions}_overlap_{int(self._overlap*100)}")

            # Read the ply file
            file_path = os.path.join(self._data_dir, ply_file)
            ply_data = PlyData.read(file_path)
            data_map = ply_data.elements[0].data

            # Split the point cloud
            tile_map = split_ply_point_cloud(data_map, self._partitions, cache_dir, overlap=self._overlap)

            # Add the tile map values to the split files list
            for _, value in tile_map.items():
                self._split_files.append(value)

        # Compute the class distribution only once
        class_distribution_file = os.path.join(self._data_dir, f"class_distribution.json")
        if not os.path.exists(class_distribution_file):
            print(f"The class distribution file {class_distribution_file} does not exist, computing it now...")
            self._class_distribution = compute_class_distribution(self)
            # Store the class distribution in a file for later use. Convert the tuple of unique, counts, labels to a dictionary
            class_distribution_dict = {
                "unique": self._class_distribution[0].tolist(),
                "counts": self._class_distribution[1].tolist(),
                "labels": self._class_distribution[2].tolist()
            }
            with open(class_distribution_file, 'w') as f:
                json.dump(class_distribution_dict, f)
        else:
            with open(os.path.join(self._root, f"class_distribution.json"), 'r') as f:
                class_distribution_dict = json.load(f)
                self._class_distribution = (np.array(class_distribution_dict["unique"]), np.array(class_distribution_dict["counts"]), np.array(class_distribution_dict["labels"]))

        # This should be done outside of the dataset
        self.labelweights = None
        if weight_type == 'Sklearn':
            if self._split != 'test':
                unique_labels, _, all_labels = self._class_distribution
                self.labelweights = np.float32(compute_class_weight(class_weight="balanced", classes=unique_labels, y=all_labels))
            else:
                self.labelweights = None
        if weight_type == 'EffectiveNumSamples':
            if self._split != 'test':
                unique_labels, counts, _ = self._class_distribution
                effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
                self.labelweights = 1 / effective_num
            else:
                self.labelweights = None

        elif weight_type == 'Custom':
            raise NotImplementedError("Custom weights are not implemented yet")            

    def get_class_distribution(self) -> tuple:
        return self._class_distribution

    def __len__(self):
        return len(self._split_files)

    def __getitem__(self, idx):
        # Read a tile txt file
        txt_file = self._split_files[idx]

        # Read the txt file
        data = np.loadtxt(txt_file)

        if self._intensity:
            points = data[:, :4]  # Keep only the first 4 columns (x,y,z,intensity)
        else:
            points = data[:, :3]

        # Normalize Point Cloud to (0, 1)
        points = normalize_points(points, normalize=self._normalize)

        # Extract the labels, which is the 5th column
        targets = data[:, 4]

        # Downsample point cloud
        if self._npoints:
            print(f"Downsampling point cloud to {self._npoints} points with method {self._downsampling_method}...")
            if self._downsampling_method == 'planar_aware':
                points, targets = downsample_inverse_planar_aware(points, targets, npoints=self._npoints)
            elif self._downsampling_method == 'uniform':
                points, targets = downsample(points, targets, npoints=self._npoints)
            elif self._downsampling_method == 'feature_based':
                points, targets = downsample_feature_based(points, targets, npoints=self._npoints)
            elif self._downsampling_method == 'biometric':
                points, targets = downsample_biometric(points, targets, npoints=self._npoints)
            elif self._downsampling_method == 'combined':
                points, targets = downsample_combined(points, targets, npoints=self._npoints)
            elif self._downsampling_method =='curvature':
                points, targets = downsample_curvature_based(points, targets, npoints=self._npoints)
            elif self._downsampling_method == 'test':
                points, targets = downsample_test(points, targets, npoints=self._npoints)
            elif self._downsampling_method == 'parallel_combined':
                points, targets = downsample_parallel_combined(points, targets, npoints=self._npoints)
            else:
                raise ValueError(f"Unknown downsampling method {self._downsampling_method}")
        # Convert to tensor
        print(f"The shape of points is {points.shape} and labels is {targets.shape}")
        points = torch.tensor(points, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)

        # The targets are indexed from 1, so we need to subtract 1
        targets -= 1

        return points, targets



    def get_categories(self):
        return self.CATEGORIES

def split_ply_point_cloud(data_map : np.memmap, N : int, cache_path : str = 'cache', overlap : float = 0.0) -> dict:
    # Create the tile map dictionary
    tile_map = {}
    tile_indices = get_all_tile_indices(N)
    for tile_idx in tile_indices:
        split_file = os.path.join(cache_path,
                                 f"tile_{tile_idx[0]}_{tile_idx[1]}.txt")
        tile_map.update({tile_idx : split_file})

    # Check if the cache directory exists already
    if os.path.exists(cache_path):
        # Check if the files already exist
        all_files_exist = True
        for _, value in tile_map.items():
            if not os.path.exists(value):
                all_files_exist = False
                break
        if all_files_exist:
            print("This split already exists in cache, skipping.")
            return tile_map

    # Create the cache directory if it does not exist
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # Create a lock map to protect access to the tile files
    x_min, y_min, x_interval, y_interval = calculate_bounds_and_intervals(data_map, N)

    # Split point cloud into chunks for parallel processing
    num_chunks = cpu_count()
    chunks = np.array_split(data_map, num_chunks)

    # Create the map of locks
    lock_map = {}
    for key, _ in tile_map.items():
        lock_map[key] = Lock()

    partial_process_chunk = partial(process_chunk, txt_map=tile_map, overlap=overlap)

    with Pool(num_chunks, initializer=init_locks, initargs=(lock_map,)) as pool:
        pool.starmap(partial_process_chunk,
                    [(chunk, x_min, y_min, x_interval, y_interval, N) for chunk in chunks])

    return tile_map

def calculate_bounds_and_intervals(data_map : np.memmap, N : int):
    """
        This function takes a point cloud and N as input and returns the
        minimum and maximum values for x and y and the intervals for x and y
        to split the point cloud into N x N tiles.
        @param point_cloud: The point cloud to split
        @param N: The number of tiles to split the point cloud into
        @return: The minimum and maximum values for x and y and the intervals for x and y 
    """
    x_min, x_max = np.min(data_map['x']), np.max(data_map['x'])
    y_min, y_max = np.min(data_map['y']), np.max(data_map['y'])

    x_interval = (x_max - x_min) / N
    y_interval = (y_max - y_min) / N

    return x_min, y_min, x_interval, y_interval

def process_chunk(chunk : np.memmap,
                  x_min : float, y_min : float, x_interval : float, y_interval : float, N : int,
                  txt_map : dict, overlap : float = 0.0):
    """
        This function processes a chunk of a point cloud and splits it into tiles.
        @param chunk: The chunk of the point cloud to process
        @param x_min: The minimum value for x
        @param y_min: The minimum value for y
        @param x_interval: The interval for x
        @param y_interval: The interval for y
        @param N: The number of tiles to split the point cloud into
    """
    tiles = {}
    for point in chunk:
        x, y = point['x'], point['y']
        tiles_point = get_tile(x, y, x_min, y_min, x_interval, y_interval, N, overlap=overlap)
        for tile in tiles_point:
            if tile not in tiles:
                tiles[tile] = []
            tiles[tile].append(point)

    # Store the tiles with text files
    for tile, points in tiles.items():
        # Critical section to write to the txt file
        # This must be protected with a lock
        with lock_map[tile]:
            with open(txt_map[tile], 'a') as f:
                for point in points:
                    f.write(f"{point['x']} {point['y']} {point['z']} {point['intensity']} {point['sem_class']} {point['ins_class']}\n")

    return None

def get_all_tile_indices(N: int) -> list:
    """
    This function generates all possible tile indices for a given number of divisions.
    @param N: The number of tiles to split the point cloud into
    @return: A list of tuples representing all possible tile indices
    """
    tile_indices = [(i, j) for i in range(N) for j in range(N)]
    return tile_indices

# Function to determine the tile of a point
def get_tile(x : float, y : float, x_min : float, y_min : float, x_interval : float, y_interval : float, N : int, overlap : float = 0.0) -> tuple:
    """
        This function takes a point, the bounds, the intervals and the number of tiles
        and returns the tile the point belongs to.
        @param x: The x coordinate of the point
        @param y: The y coordinate of the point
        @param x_min: The minimum value for x
        @param y_min: The minimum value for y
        @param x_interval: The interval for x
        @param y_interval: The interval for y
        @param N: The number of tiles to split the point cloud into
        @param overlap: The overlap between tiles, with overlap a point can belong to multiple tiles
        @return: The tile the point belongs to
    """
    # List to store the tiles the point belongs to
    tiles = []

    # Calculate the extended interval considering the overlap
    x_overlap_interval = x_interval * overlap
    y_overlap_interval = y_interval * overlap

    # Determine the potential range of tile indices the point might belong to
    x_start_index = int((x - x_min) / x_interval)
    y_start_index = int((y - y_min) / y_interval)

    if (x_start_index == N or y_start_index == N):
        tiles.append((max(0,x_start_index-1), max(0,y_start_index-1)))
    else:
        # Iterate through potential tiles the point could belong to
        for i in range(x_start_index - 1, x_start_index + 2):
            if i < 0 or i >= N:
                continue
            for j in range(y_start_index - 1, y_start_index + 2):
                if j < 0 or j >= N:
                    continue
                # Calculate the bounds of the current tile
                x_quad_min = (x_min + i * x_interval) - x_overlap_interval
                x_quad_max = (x_min + i * x_interval) + x_interval + x_overlap_interval
                y_quad_min = (y_min + j * y_interval) - y_overlap_interval
                y_quad_max = (y_min + j * y_interval) + y_interval + y_overlap_interval

                # Check if the point lies within the current tile bounds
                if x_quad_min <= x < x_quad_max and y_quad_min <= y < y_quad_max:
                    tiles.append((i, j))

    # Filter out tiles that are outside the valid range [0, N-1]
    valid_tiles = []
    for i, j in tiles:
        if 0 <= i < N and 0 <= j < N:
            valid_tiles.append((i, j))
        else:
            print(f"Skipping point ({x}, {y}) in tile ({i}, {j}) as it is outside the valid range [0, {N-1}]")
            print(f"The method was called with x_min={x_min}, y_min={y_min}, x_interval={x_interval}, y_interval={y_interval}, N={N}, overlap={overlap}")
            print(f"Start index was ({x_start_index}, {y_start_index})")

    return valid_tiles

def visualize_pointcloud(point_cloud, labels, window_name : str = "DALES point cloud"):

    # Extract the data
    points = point_cloud[:, :3].numpy()
    intensity = point_cloud[:, 3].numpy() if point_cloud.shape[1] == 4 else None
    print(f"Labels are {labels} with len {len(labels)}")
    sem_class = labels.numpy()

    # Print some basic statistics for coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    print("\nStatistics for coordinates:")
    print(f"  X: min={x.min()}, max={x.max()}, mean={x.mean()}")
    print(f"  Y: min={y.min()}, max={y.max()}, mean={y.mean()}")
    print(f"  Z: min={z.min()}, max={z.max()}, mean={z.mean()}")

    # Example color palette for semantic classes
    color_palette = np.array([
        #[1, 3, 117],    # Class 0: Unknown
        [1, 79, 156],   # Class 1: Blue: Ground
        [1, 114, 3],    # Class 2: Green: Vegetation
        [222, 47, 225], # Class 3: Pink: Cars
        [237, 236, 5],  # Class 4: Yellow: Trucks
        [2, 205, 1],    # Class 5: Light Green: Power Lines
        [5, 216, 223],  # Class 6: Light Blue: Fences
        [250, 125, 0],  # Class 7: Orange: Poles
        [196 ,1, 1],    # Class 8: Red: Buildings
    ])

    # Normalize the palette to [0, 1] range for Open3D
    color_palette = color_palette / 255.0

    # Get the colors based on semantic class
    colors = color_palette[sem_class % len(color_palette)]

    if intensity is not None:
        # Normalize intensity to the range [0, 1]
        intensity = intensity.astype(np.float32)
        intensity_min = intensity.min()
        intensity_max = intensity.max()
        intensity = (intensity - intensity_min) / (intensity_max - intensity_min)

        # Calculate normals for the point cloud
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Define a light source direction
        light_direction = np.array([1.0, 1.0, 1.0])
        light_direction /= np.linalg.norm(light_direction)  # Normalize the light direction

        # Get the normals
        normals = np.asarray(o3d_point_cloud.normals)

        # Compute the shading (dot product between normals and light direction)
        shading = np.dot(normals, light_direction)
        shading = np.clip(shading, 0, 1)  # Ensure shading is within [0, 1]

        # Blend intensity with colors, giving more weight to colors
        intensity_weight = 0.25  # Adjust this to give less weight to intensity
        blended_colors = colors * (1 - intensity_weight) + colors * intensity[:, np.newaxis] * intensity_weight

        # Apply the shading effect
        shadow_effect = 0.0  # Adjust this to control shadow intensity
        blended_colors = blended_colors * ((1 - shadow_effect) + shading[:, np.newaxis] * shadow_effect)

        # Ensure colors are within [0, 1]
        blended_colors = np.clip(blended_colors, 0, 1)
    else:
        blended_colors = colors

    # Create an Open3D point cloud object
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d_point_cloud.colors = o3d.utility.Vector3dVector(blended_colors)


    # Visualize the point cloud
    o3d.visualization.draw_geometries([o3d_point_cloud],
                                       window_name=window_name)
