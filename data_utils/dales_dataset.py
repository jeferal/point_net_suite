import os

from plyfile import PlyData

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import Dataset

class DalesDataset(Dataset):

    def __init__(self, root : str, split : str, intensity : bool = False, instance_seg : bool = False):
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

        # Loop over all ply files and read the data
        cache_dir = os.path.join(self._root, "cache")
        # Create the cache directory if it does not exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        for ply_file in self._ply_files:
            # Create a directory for this ply
            ply_cache_dir = os.path.join(cache_dir, ply_file)
            if not os.path.exists(ply_cache_dir):
                os.makedirs(ply_cache_dir)

            print(f"Processing file {ply_file}")
            file_path = os.path.join(self._data_dir, ply_file)
            print(f"Reading file {file_path}")
            ply_data = PlyData.read(file_path)

            # Obtain the point cloud
            point_cloud_x = torch.from_numpy(ply_data.elements[0].data['x'])
            point_cloud_y = torch.from_numpy(ply_data.elements[0].data['y'])
            point_cloud_z = torch.from_numpy(ply_data.elements[0].data['z'])

            if self._intensity:
                intensity = torch.from_numpy(ply_data.elements[0].data['intensity'])
                point_cloud = torch.stack((point_cloud_x, point_cloud_y, point_cloud_z, intensity), dim=1)
            else:
                # Stack the point cloud coordinate x, y, z
                point_cloud = torch.stack((point_cloud_x, point_cloud_y, point_cloud_z), dim=1)

            print(f"Size of point cloud {point_cloud.size()}")

            # Partition the point cloud into N quadrants
            partitions = split_point_cloud(point_cloud, 4)

            exit()
            # Store the partitions in a .txt file
            for key, value in partitions.items():
                print(f"Quadrant {key} has {value.size(0)} points")
                np.savetxt(os.path.join(ply_cache_dir, f"quadrant_{key}.txt"), value.numpy())

    def __len__(self):
        return len(self._ply_files)

    def __getitem__(self, idx):

        # Index the ply files list
        ply_file = self._ply_files[idx]

        # Read the ply file
        file_path = os.path.join(self._data_dir, ply_file)
        print(file_path)
        ply_data = PlyData.read(file_path)

        # Obtain the point cloud
        point_cloud_x = torch.from_numpy(ply_data.elements[0].data['x'])
        point_cloud_y = torch.from_numpy(ply_data.elements[0].data['y'])
        point_cloud_z = torch.from_numpy(ply_data.elements[0].data['z'])

        if self._intensity:
            intensity = torch.from_numpy(ply_data.elements[0].data['intensity'])
            point_cloud = torch.stack((point_cloud_x, point_cloud_y, point_cloud_z, intensity), dim=1)
        else:
            # Stack the point cloud coordinate x, y, z
            point_cloud = torch.stack((point_cloud_x, point_cloud_y, point_cloud_z), dim=1)

        if self._instance_seg:
            raise NotImplementedError("Instance segmentation loading not implemented yet")

        # Obtain the labels
        labels = torch.from_numpy(ply_data.elements[0].data['sem_class'])

        return point_cloud, labels

def calculate_bounds_and_intervals(point_cloud, N):
    x_min, x_max = torch.min(point_cloud[:,0]), torch.max(point_cloud[:,0])
    y_min, y_max = torch.min(point_cloud[:,1]), torch.max(point_cloud[:,1])

    x_interval = (x_max - x_min) / N
    y_interval = (y_max - y_min) / N
    
    return x_min, y_min, x_interval, y_interval

def split_point_cloud(point_cloud, N):
    x_min, y_min, x_interval, y_interval = calculate_bounds_and_intervals(point_cloud, N)
    
    # Split point cloud into chunks for parallel processing
    num_chunks = cpu_count()
    chunks = np.array_split(point_cloud, num_chunks)
    
    with Pool(processes=num_chunks) as pool:
        print(f"Processing {num_chunks} chunks")
        results = pool.starmap(process_chunk, [(chunk, x_min, y_min, x_interval, y_interval, N) for chunk in chunks])
    print(f"Results: {results}")
    breakpoint()

    quadrants = merge_results(results)

    return quadrants

def merge_results(results):
    final_quadrants = {}
    for result in results:
        for quadrant, points in result.items():
            if quadrant not in final_quadrants:
                final_quadrants[quadrant] = []
            final_quadrants[quadrant].extend(points)
    return final_quadrants

def process_chunk(chunk, x_min, y_min, x_interval, y_interval, N):
    quadrants = {}
    for point in tqdm(chunk):
        x, y, z = point
        quadrant = get_quadrant(x, y, x_min, y_min, x_interval, y_interval, N)
        if quadrant not in quadrants:
            quadrants[quadrant] = []
        quadrants[quadrant].append(point)
    
    print("Chunk processed")
    return quadrants

# Function to determine the quadrant of a point
def get_quadrant(x, y, x_min, y_min, x_interval, y_interval, N) -> tuple:
    x_index = int((x - x_min) / x_interval)
    y_index = int((y - y_min) / y_interval)

    # Ensure the index is within the bounds
    x_index = min(x_index, N - 1)
    y_index = min(y_index, N - 1)

    return (x_index, y_index)
