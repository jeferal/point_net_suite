import os

from plyfile import PlyData

import open3d as o3d

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from multiprocessing import Lock
from functools import partial

import torch
from torch.utils.data import Dataset

def init_locks(l):
    # Define as many locks as the number of quadrants
    # And store them in a dictionary
    global lock_map
    lock_map = l

class DalesDataset(Dataset):

    def __init__(self, root : str, split : str, partitions = 1, intensity : bool = False, instance_seg : bool = False):
        self._root = root
        self._split = split

        # Create the data directory
        self._data_dir = os.path.join(self._root, self._split)
        self._cache_dir = os.path.join(self._root, self._split, "cache")

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

        self._partitions = partitions

        # Quadrants indices are the same for all ply files
        quadrant_indices = get_all_quadrant_indices(self._partitions)

        # List with the path of the txt files
        self._split_files = []
        # Loop for every file
        for ply_file in tqdm(self._ply_files):
            # Create a dictionary to map quadrant indices to path of the quadrant txt file
            self._quadrant_map = {}
            for quadrant_idx in enumerate(quadrant_indices):
                split_file = os.path.join(self._cache_dir,
                                          os.path.splitext(ply_file)[0],
                                          f"quadrant_{quadrant_idx[0]}_{quadrant_idx[1]}.txt")
                self._split_files.append(split_file)
                self._quadrant_map[quadrant_idx] = split_file

            # Read the ply file
            file_path = os.path.join(self._data_dir, ply_file)
            ply_data = PlyData.read(file_path)
            data_map = ply_data.elements[0].data

            # Split the point cloud
            if not self.does_cache_exist(ply_file):
                print(f"Splitting the point cloud {ply_file}")
                split_ply_point_cloud(data_map, self._partitions, self._quadrant_map)

    def does_cache_exist(self, ply_file : str):
        # TODO: This function could check if every file exists
        ply_cache_dir = os.path.join(self._cache_dir, os.path.splitext(ply_file)[0])
        return os.path.exists(ply_cache_dir)

    def __len__(self):
        return len(self._ply_files * self._partitions)

    def __getitem__(self, idx):
        # Read a quadrant txt file
        txt_file = self._split_files[idx]
        print(f"Reading this file {txt_file}")
        # Read the txt file
        data = np.loadtxt(txt_file)

        # Convert to a tensor Nx4
        data = torch.from_numpy(data).float()
        # TODO: Write labels
        labels = torch.zeros(data.shape[0])

        return data, labels

def split_ply_point_cloud(data_map : np.memmap, N : int, quadrant_map : dict) -> None:
    print(quadrant_map.keys())
    
    # Create a lock map to protect access to the quadrant files
    x_min, y_min, x_interval, y_interval = calculate_bounds_and_intervals(data_map, N)

    # Split point cloud into chunks for parallel processing
    num_chunks = cpu_count()
    chunks = np.array_split(data_map, num_chunks)

    # Create the map of locks
    lock_map = {}
    for key, _ in quadrant_map.items():
        lock_map[key] = Lock()

    partial_process_chunk = partial(process_chunk, txt_map=quadrant_map)

    print(f"Processing {num_chunks} chunks")
    with Pool(num_chunks, initializer=init_locks, initargs=(lock_map,)) as pool:
        pool.starmap(partial_process_chunk,
                    [(chunk, x_min, y_min, x_interval, y_interval, N) for chunk in chunks])

    return None

def calculate_bounds_and_intervals(data_map : np.memmap, N : int):
    """
        This function takes a point cloud and N as input and returns the
        minimum and maximum values for x and y and the intervals for x and y
        to split the point cloud into N x N quadrants.
        @param point_cloud: The point cloud to split
        @param N: The number of quadrants to split the point cloud into
        @return: The minimum and maximum values for x and y and the intervals for x and y 
    """
    x_min, x_max = np.min(data_map['x']), np.max(data_map['x'])
    y_min, y_max = np.min(data_map['y']), np.max(data_map['y'])

    x_interval = (x_max - x_min) / N
    y_interval = (y_max - y_min) / N

    return x_min, y_min, x_interval, y_interval


def split_point_cloud_open3d(input_pc, N):
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    # Assign the numpy array to the Open3D point cloud object
    point_cloud.points = o3d.utility.Vector3dVector(input_pc.numpy())

    # Crop the point cloud
    x_min, y_min, x_interval, y_interval = calculate_bounds_and_intervals(input_pc, N)

    # Crop the point cloud
    for i in range(N):
        for j in range(N):
            print(f"Processing quadrant {i}, {j}")
            # Create a bounding box
            x_min_bound = x_min + i * x_interval
            x_max_bound = x_min + (i + 1) * x_interval
            y_min_bound = y_min + j * y_interval
            y_max_bound = y_min + (j + 1) * y_interval
            # Create a bounding box
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(x_min_bound, y_min_bound, -np.inf), max_bound=(x_max_bound, y_max_bound, np.inf))
            # Crop the point cloud
            cropped_point_cloud = point_cloud.crop(bounding_box)
            print(f"Point cloud has this shape {np.asarray(cropped_point_cloud.points).shape}")
            # Visualize the point cloud
            o3d.visualization.draw_geometries([cropped_point_cloud], window_name=f"Quadrant {i}, {j}") 

def process_chunk(chunk : np.memmap,
                  x_min : float, y_min : float, x_interval : float, y_interval : float, N : int,
                  txt_map : dict):
    """
        This function processes a chunk of a point cloud and splits it into quadrants.
        @param chunk: The chunk of the point cloud to process
        @param x_min: The minimum value for x
        @param y_min: The minimum value for y
        @param x_interval: The interval for x
        @param y_interval: The interval for y
        @param N: The number of quadrants to split the point cloud into
    """
    quadrants = {}
    for point in chunk:
        x, y = point['x'], point['y']
        quadrant = get_quadrant(x, y, x_min, y_min, x_interval, y_interval, N)
        if quadrant not in quadrants:
            quadrants[quadrant] = []
        quadrants[quadrant].append(point)

    # Store the quadrants with text files
    for quadrant, points in quadrants.items():
        # Critical section to write to the txt file
        # This must be protected with a lock
        with lock_map[quadrant]:
            # Create the base directory if it does not exist
            if not os.path.exists(os.path.dirname(txt_map[quadrant])):
                os.makedirs(os.path.dirname(txt_map[quadrant]))
            with open(txt_map[quadrant], 'a') as f:
                for point in points:
                    f.write(f"{point['x']} {point['y']} {point['z']} {point['intensity']} {point['sem_class']} {point['ins_class']}\n")

    return None

def get_all_quadrant_indices(N: int) -> list:
    """
    This function generates all possible quadrant indices for a given number of divisions.
    @param N: The number of quadrants to split the point cloud into
    @return: A list of tuples representing all possible quadrant indices
    """
    quadrant_indices = [(i, j) for i in range(N) for j in range(N)]
    return quadrant_indices

# Function to determine the quadrant of a point
def get_quadrant(x : float, y : float, x_min : float, y_min : float, x_interval : float, y_interval : float, N : int) -> tuple:
    """
        This function takes a point, the bounds, the intervals and the number of quadrants
        and returns the quadrant the point belongs to.
        @param x: The x coordinate of the point
        @param y: The y coordinate of the point
        @param x_min: The minimum value for x
        @param y_min: The minimum value for y
        @param x_interval: The interval for x
        @param y_interval: The interval for y
        @param N: The number of quadrants to split the point cloud into
        @return: The quadrant the point belongs to
    """
    x_index = int((x - x_min) / x_interval)
    y_index = int((y - y_min) / y_interval)

    # Ensure the index is within the bounds
    x_index = min(x_index, N - 1)
    y_index = min(y_index, N - 1)

    return (x_index, y_index)
