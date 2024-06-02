import os

from plyfile import PlyData

import open3d as o3d

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
from torch.utils.data import Dataset

class DalesDataset(Dataset):

    def __init__(self, root : str, split : str, partitions = 1, intensity : bool = False, instance_seg : bool = False):
        self._root = root
        self._split = split

        # Create the data directory
        self._data_dir = os.path.join(self._root, self._split)

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

        # Loop for every file
        for ply_file in self._ply_files:
            print(f"Processing file {ply_file}")
            # Read the ply file
            file_path = os.path.join(self._data_dir, ply_file)
            ply_data = PlyData.read(file_path)

            data_map = ply_data.elements[0].data

            # Split the point cloud
            split_ply_point_cloud(data_map, self._partitions)

            exit()

    def __len__(self):
        return len(self._ply_files * self._partitions)

    def __getitem__(self, idx):
        # TODO: Implement the get item function
        return

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

def process_chunk(chunk : np.memmap, x_min : float, y_min : float, x_interval : float, y_interval : float, N : int):
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
    for point in tqdm(chunk):
        x, y = point['x'], point['y']
        quadrant = get_quadrant(x, y, x_min, y_min, x_interval, y_interval, N)
        if quadrant not in quadrants:
            quadrants[quadrant] = []
        quadrants[quadrant].append(point)
    
    print("Chunk processed")
    return None


def split_ply_point_cloud(data_map : np.memmap, N : int) -> None:
    x_min, y_min, x_interval, y_interval = calculate_bounds_and_intervals(data_map, N)

    print(f"Got these values: {x_min}, {y_min}, {x_interval}, {y_interval}")

    # Split point cloud into chunks for parallel processing
    num_chunks = cpu_count()
    chunks = np.array_split(data_map, num_chunks)

    # Show the shape of the chunks
    for chunk in chunks:
        print(f"Chunk shape: {chunk.shape}")

    print(f"Splitting point cloud into {num_chunks} chunks")

    # Create a partial function to pass the arguments that do not change
    with Pool(num_chunks) as pool:
        pool.starmap(process_chunk,
                     [(chunk, x_min, y_min, x_interval, y_interval, N) for chunk in chunks])

    exit()
    return None

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
