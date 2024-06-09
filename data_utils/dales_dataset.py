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

    def __init__(self, root : str, split : str, partitions = 1, intensity : bool = False, instance_seg : bool = False, overlap : float = 0.0):
        self._root = root
        self._split = split

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

        self._partitions = partitions

        # Quadrants indices are the same for all ply files
        quadrant_indices = get_all_quadrant_indices(self._partitions)

        # List with the path of the txt files
        self._split_files = []
        # Loop for every file
        for ply_file in tqdm(self._ply_files):

            # Cache path
            cache_dir = os.path.join(self._cache_dir,
                                     f"{os.path.splitext(ply_file)[0]}_{self._partitions}_overlap_{self._overlap*100}")

            # Read the ply file
            file_path = os.path.join(self._data_dir, ply_file)
            ply_data = PlyData.read(file_path)
            data_map = ply_data.elements[0].data

            # Split the point cloud
            tile_map = split_ply_point_cloud(data_map, self._partitions, cache_dir, overlap=self._overlap)

            # Add the tile map values to the split files list
            for _, value in tile_map.items():
                self._split_files.append(value)

    def __len__(self):
        return len(self._ply_files * self._partitions)

    def __getitem__(self, idx):
        # Read a quadrant txt file
        txt_file = self._split_files[idx]
        print(f"Reading this file {txt_file}")
        # Read the txt file
        data = np.loadtxt(txt_file)

        # Convert to a tensor Nx4
        data = torch.from_numpy(data).double()
        # TODO: Write labels
        labels = torch.zeros(data.shape[0])

        return data, labels

def split_ply_point_cloud(data_map : np.memmap, N : int, cache_path : str = 'cache', overlap : float = 0.0) -> dict:
    # Create the tile map dictionary
    tile_map = {}
    tile_indices = get_all_quadrant_indices(N)
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

    # Create a lock map to protect access to the quadrant files
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

def process_chunk(chunk : np.memmap,
                  x_min : float, y_min : float, x_interval : float, y_interval : float, N : int,
                  txt_map : dict, overlap : float = 0.0):
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
        quadrants_point = get_quadrant(x, y, x_min, y_min, x_interval, y_interval, N, overlap=overlap)
        for quadrant in quadrants_point:
            if quadrant not in quadrants:
                quadrants[quadrant] = []
            quadrants[quadrant].append(point)

    # Store the quadrants with text files
    for quadrant, points in quadrants.items():
        # Critical section to write to the txt file
        # This must be protected with a lock
        with lock_map[quadrant]:
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
def get_quadrant(x : float, y : float, x_min : float, y_min : float, x_interval : float, y_interval : float, N : int, overlap : float = 0.0) -> tuple:
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
        @param overlap: The overlap between quadrants, with overlap a point can belong to multiple quadrants
        @return: The quadrant the point belongs to
    """
    # List to store the quadrants the point belongs to
    quadrants = []

    # Calculate the extended interval considering the overlap
    x_overlap_interval = x_interval * overlap
    y_overlap_interval = y_interval * overlap

    # Determine the potential range of quadrant indices the point might belong to
    x_start_index = int((x - x_min) / x_interval)
    y_start_index = int((y - y_min) / y_interval)

    if (x_start_index == N or y_start_index == N):
        quadrants.append((max(0,x_start_index-1), max(0,y_start_index-1)))
    else:
        # Iterate through potential quadrants the point could belong to
        for i in range(x_start_index - 1, x_start_index + 2):
            if i < 0 or i >= N:
                continue
            for j in range(y_start_index - 1, y_start_index + 2):
                if j < 0 or j >= N:
                    continue
                # Calculate the bounds of the current quadrant
                x_quad_min = (x_min + i * x_interval) - x_overlap_interval
                x_quad_max = (x_min + i * x_interval) + x_interval + x_overlap_interval
                y_quad_min = (y_min + j * y_interval) - y_overlap_interval
                y_quad_max = (y_min + j * y_interval) + y_interval + y_overlap_interval

                # Check if the point lies within the current quadrant bounds
                if x_quad_min <= x < x_quad_max and y_quad_min <= y < y_quad_max:
                    quadrants.append((i, j))

    # Filter out quadrants that are outside the valid range [0, N-1]
    valid_quadrants = []
    for i, j in quadrants:
        if 0 <= i < N and 0 <= j < N:
            valid_quadrants.append((i, j))
        else:
            print(f"Skipping point ({x}, {y}) in quadrant ({i}, {j}) as it is outside the valid range [0, {N-1}]")
            print(f"The method was called with x_min={x_min}, y_min={y_min}, x_interval={x_interval}, y_interval={y_interval}, N={N}, overlap={overlap}")
            print(f"Start index was ({x_start_index}, {y_start_index})")

    return valid_quadrants
