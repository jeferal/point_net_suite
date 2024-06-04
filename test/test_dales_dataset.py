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

    # TODO: Simplify the test with a point cloud that is not that big
    # Generate random points and split them into N x N splits

    # TODO: Add a test for how we split the x,y coordinates given N

    # TODO: Add a test when we add the 'overlap' feature

    def test_quadrant_indices(self):
        # This method tests the get_all_quadrant_indices method
        # by checking that the number of indices is correct
        # and that the indices are unique

        n = 5
        quadrant_indices = get_all_quadrant_indices(n)

        # Check that the number of indices is correct
        assert len(quadrant_indices) == n ** 2

        # Check that the indices are unique
        assert len(set(quadrant_indices)) == len(quadrant_indices)

        correct_result = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

        assert quadrant_indices == correct_result

    def test_split(self):
        # This method splits a ply point cloud into N x N splits and
        # checks that the resulting point cloud added are the same as the original

        # Load a ply
        ply_file = os.path.join('data', 'DALESObjects', 'train', '5080_54435_new.ply')
        ply_data = PlyData.read(ply_file)
        data_map = ply_data.elements[0].data

        n = 30
        quadrant_indices = get_all_quadrant_indices(n)
        cache_dir = os.path.join('data', 'DALESObjects', 'train', 'test_cache')

        # Remove the cache directory if it exists
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)

        quadrant_map = {}
        for quadrant_idx in quadrant_indices:
            print(quadrant_idx)
            split_file = os.path.join(cache_dir,
                                     '5080_54435_new',
                                     f"quadrant_{quadrant_idx[0]}_{quadrant_idx[1]}.txt")
            quadrant_map[quadrant_idx] = split_file

        print(f"Len of data map: {len(data_map)}")
        split_ply_point_cloud(data_map, n, quadrant_map)

        # Check that the cache directory was created
        assert os.path.exists(cache_dir)

        # Check that the split files were created with the proper name
        for quadrant_idx in quadrant_indices:
            assert os.path.exists(quadrant_map[quadrant_idx])

        # Load all the points from the split files and concatenate them in 
        # a single numpy array with x, y, z, intensity, sem_class, ins_class
        # split points should be a numpy array with shape (n, 6)
        split_points = np.empty((0, 6))
        for quadrant_idx in quadrant_indices:
            split_file = quadrant_map[quadrant_idx]
            split_data = np.loadtxt(split_file)
            print(f"Got split data with shape: {split_data.shape}")
            split_points = np.concatenate((split_points, split_data), axis=0)

        print(f"Split points shape: {split_points.shape}")

        assert split_points.shape[0] == data_map.shape[0]
        assert split_points.shape[1] == len(data_map[0])

        # Check that both point cloud are the same
        print("The point clouds have the same dimensionality")
        print("Comparing the point clouds...")
        compare_point_clouds(data_map, split_points)

        print("The point clouds are the same")


if __name__ == '__main__':
    unittest.main()