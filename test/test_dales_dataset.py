import unittest

import os

from plyfile import PlyData
import numpy as np

from data_utils.dales_dataset import DalesDataset, split_ply_point_cloud, get_all_quadrant_indices


def compare_point_clouds(pc1, pc2):
    """
    Compare two point clouds.
    """
    pc1_sorted = np.array(sorted(pc1, key=lambda x: (x[0], x[1])))
    pc2_sorted = np.array(sorted(pc2, key=lambda x: (x[0], x[1])))

    # Compare the sorted arrays
    return np.array_equal(pc1_sorted, pc2_sorted)


class TestDalesDataset(unittest.TestCase):

    def test_split(self):
        # This method splits a ply point cloud into N x N splits and
        # checks that the resulting point cloud added are the same as the original

        # Load a ply
        ply_file = os.path.join('data', 'DALESObjects', 'train', '5080_54435_new.ply')
        ply_data = PlyData.read(ply_file)
        data_map = ply_data.elements[0].data

        partitions = 13

        quadrant_indices = get_all_quadrant_indices(partitions)
        cache_dir = os.path.join('data', 'DALESObjects', 'train', 'test_cache')

        # Remove the cache directory if it exists
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)

        quadrant_map = {}
        for i, index in enumerate(quadrant_indices):
            split_file = os.path.join(cache_dir,
                                     '5080_54435_new',
                                     f"partition_{partitions}_{i}.txt")
            print(split_file)
            quadrant_map[index] = split_file

        print(f"Len of data map: {len(data_map)}")
        split_ply_point_cloud(data_map, 13, quadrant_map)

        # Load the split point clouds and concatenate them into a single numpy array
        split_point_clouds = []
        for i in range(partitions):
            split_file = os.path.join(cache_dir,
                                     os.path.splitext(ply_file)[0],
                                     f"partition_{partitions}_{i}.txt")
            split_data = np.loadtxt(split_file)
            split_point_clouds.append(split_data)
        
        split_point_clouds = np.concatenate(split_point_clouds)
        print(split_point_clouds.shape)

        assert len(data_map) == len(split_point_clouds)

if __name__ == '__main__':
    unittest.main()