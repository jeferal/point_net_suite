import os

from plyfile import PlyData
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
from plyfile import PlyData
import matplotlib.pyplot as plt

class DalesDataset(Dataset):

    def __init__(self, root : str, split : str, intensity : bool = True, instance_seg : bool = False):
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

    def __len__(self):
        return len(self._ply_files)

    def __getitem__(self, idx):

        # Index the ply files list
        ply_file = self._ply_files[idx]

        # Read the ply file
        file_path = os.path.join(self._data_dir, ply_file)
        print(file_path)
        ply_data = PlyData.read(file_path)
        print("Available attributes:", ply_data.elements[0].data.dtype.names)

        # Obtain the point cloud
        point_cloud_x = torch.from_numpy(ply_data.elements[0].data['x'])
        point_cloud_y = torch.from_numpy(ply_data.elements[0].data['y'])
        point_cloud_z = torch.from_numpy(ply_data.elements[0].data['z'])

        print(f"point_cloud_x shape: {point_cloud_x.shape}")
        print(f"point_cloud_y shape: {point_cloud_y.shape}")
        print(f"point_cloud_z shape: {point_cloud_z.shape}")

        print("Names in ply_data", ply_data.elements[0].data.dtype.names) 
        print("self intensity", self._intensity)

        if self._intensity and 'intensity' in ply_data.elements[0].data.dtype.names:
            intensity = torch.from_numpy(ply_data.elements[0].data['intensity'])
            print(f"intensity shape: {intensity.shape}")
            print(f"intensity values: {intensity[:10]}")
            point_cloud = torch.stack((point_cloud_x, point_cloud_y, point_cloud_z, intensity), dim=1)
            
        else:
            # Stack the point cloud coordinate x, y, z
            point_cloud = torch.stack((point_cloud_x, point_cloud_y, point_cloud_z), dim=1)
            if self._intensity:
                print("Warning: Intensity flag is set, but intensity data is not available in the file.")

        print(f"point_cloud shape after stacking: {point_cloud.shape}")

        if self._instance_seg:
            raise NotImplementedError("Instance segmentation loading not implemented yet")

        # Obtain the labels
        labels = torch.from_numpy(ply_data.elements[0].data['sem_class'])
        print(f"labels shape: {labels.shape}")

        return point_cloud, labels
    
def visualize_pointcloud(point_cloud, labels):

    # Extract the data
    points = point_cloud[:, :3].numpy()
    intensity = point_cloud[:, 3].numpy() if point_cloud.shape[1] == 4 else None
    sem_class = labels.numpy()

    # Print some basic statistics for coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    print("\nStatistics for coordinates:")
    print(f"  X: min={x.min()}, max={x.max()}, mean={x.mean()}")
    print(f"  Y: min={y.min()}, max={y.max()}, mean={y.mean()}")
    print(f"  Z: min={z.min()}, max={z.max()}, mean={z.mean()}")


    # Print the first few points for inspection
    # Print the first few points for inspection
    print("\nFirst few points:")
    for i in range(min(10, len(points))):
        print(points[i], intensity[i] if intensity is not None else "-", sem_class[i])

    # Example color palette for semantic classes
    color_palette = np.array([
        [1, 3, 117],    # Class 0: Unknown
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
    o3d.visualization.draw_geometries([o3d_point_cloud], window_name="DALES point cloud")