import os

from plyfile import PlyData

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
