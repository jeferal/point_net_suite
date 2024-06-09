import unittest

import os
import shutil

import numpy as np

from data_utils.dales_dataset import split_ply_point_cloud, get_all_tile_indices, calculate_bounds_and_intervals, get_tile


def create_random_point_cloud(num_points, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Create a random point cloud with N points.
    """
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
             ('intensity', np.int32), ('sem_class', np.int32), ('ins_class', np.int32)]

    # Create the memmap array
    filename = 'random_data.dat'
    shape = (num_points,)
    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

    memmap_array['x'] = np.random.uniform(low=x_min, high=x_max, size=shape)
    memmap_array['y'] = np.random.uniform(low=y_min, high=y_max, size=shape)
    memmap_array['z'] = np.random.uniform(low=z_min, high=z_max, size=shape)
    memmap_array['intensity'] = np.random.uniform(low=0, high=1000, size=shape)
    memmap_array['sem_class'] = np.random.uniform(low=0, high=8, size=shape)
    memmap_array['ins_class'] = np.random.uniform(low=0, high=1000, size=shape)

    return memmap_array


class TestDalesDataset(unittest.TestCase):

    # TODO: Simplify the test with a point cloud that is not that big
    # Generate random points and split them into N x N splits

    # TODO: Add a test for how we split the x,y coordinates given N

    # TODO: Add a test when we add the 'overlap' feature
    def test_get_tile(self):
        x = 7
        y = 0

        x_min = 0
        y_min = 0
        x_interval = 4
        y_interval = 4
        N = 5

        # Test the first tile
        tiles = get_tile(x, y, x_min, y_min, x_interval, y_interval, N)

        self.assertEqual(tiles, [(1, 0)])

        overlap = 1.0
        tiles = get_tile(x, y, x_min, y_min, x_interval, y_interval, N, overlap)
        self.assertEqual(tiles, [(0,0), (0, 1), (1,0), (1,1), (2,0), (2,1)])

        x = 6.128175
        y = 1.6596304
        x_min = -10
        y_min = -5
        x_interval = 2
        y_interval = 1
        N = 10
        overlap = 0.0
        tiles = get_tile(x, y, x_min, y_min, x_interval, y_interval, N, overlap)
        self.assertEqual(len(tiles), 1)

        x = 3.3249833583831787
        y = 4.999974727630614
        x_min = -9.999513626098633
        y_min = -4.999854564666748
        x_interval = 1.9999460220336913
        y_interval = 0.9999829292297363
        N = 10
        overlap = 0.0
        tiles = get_tile(x, y, x_min, y_min, x_interval, y_interval, N, overlap)
        self.assertEqual(len(tiles), 1)

    def test_calculate_bounds_and_intervals(self):
        np.random.seed(42)
        # Create a point cloud (x,y,z) with N random points
        # from x_min to x_max, y_min to y_max and z_min to z_max
        # and check that the bounds and intervals are correct
        x_max, x_min = 10, -10
        y_max, y_min = 5, -5
        z_max, z_min = 10, -10
        num_points = 1000

        data_map = create_random_point_cloud(num_points,
                                             x_min, x_max,
                                             y_min, y_max,
                                             z_min, z_max)

        N = 5
        x_min, y_min, x_interval, y_interval = calculate_bounds_and_intervals(data_map, N)

        self.assertAlmostEqual(x_min, -10, delta=0.1)
        self.assertAlmostEqual(y_min, -5, delta=0.1)
        self.assertAlmostEqual(x_interval, 4, delta=0.1)
        self.assertAlmostEqual(y_interval, 2, delta=0.1)

        if os.path.exists(data_map.filename):
            os.remove(data_map.filename)

    def test_tile_indices(self):
        # This method tests the get_all_tile_indices method
        # by checking that the number of indices is correct
        # and that the indices are unique
        n = 5
        tile_indices = get_all_tile_indices(n)

        # Check that the number of indices is correct
        self.assertEqual(len(tile_indices), n ** 2)

        # Check that the indices are unique
        self.assertEqual(len(set(tile_indices)), len(tile_indices))

        correct_result = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

        self.assertEqual(tile_indices, correct_result)

    def test_split(self):
        np.random.seed(4)
        # This method splits a ply point cloud into N x N splits and
        # checks that the resulting point cloud added are the same as the original
        # Create a point cloud (x,y,z) with N random points
        # from x_min to x_max, y_min to y_max and z_min to z_max
        # and check that the bounds and intervals are correct
        x_max, x_min = 10, -10
        y_max, y_min = 5, -5
        z_max, z_min = 10, -10
        num_points = 100000

        data_map = create_random_point_cloud(num_points,
                                             x_min, x_max,
                                             y_min, y_max,
                                             z_min, z_max)

        n = 10
        tile_indices = get_all_tile_indices(n)
        cache_dir = os.path.join('test_cache')

        # Remove the cache directory if it exists
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        tile_map = split_ply_point_cloud(data_map, n, cache_dir)

        # Check that the cache directory was created
        self.assertTrue(os.path.exists(cache_dir))

        # Check that the split files were created with the proper name
        for tile_idx in tile_indices:
            self.assertTrue(os.path.exists(tile_map[tile_idx]))

        # Load all the points from the split files and concatenate them in 
        # a single numpy array with x, y, z, intensity, sem_class, ins_class
        # split points should be a numpy array with shape (n, 6)
        split_points = np.empty((0, 6))
        for tile_idx in tile_indices:
            split_file = tile_map[tile_idx]
            split_data = np.loadtxt(split_file)
            split_data = np.atleast_2d(split_data)
            split_points = np.concatenate((split_points, split_data), axis=0)

        # Assert that the number of points is the same
        # And also the number of columns (x,y,z,intensity,sem_class,ins_class)  
        self.assertEqual(split_points.shape[0], data_map.shape[0])
        self.assertEqual(split_points.shape[1], len(data_map[0]))

        # Check that both point cloud are the same
        self.compare_point_clouds(data_map, split_points)

        if os.path.exists(data_map.filename):
            os.remove(data_map.filename)

    def compare_point_clouds(self, pc1, pc2):
        """
        Compare two point clouds.
        """
        pc1_sorted = np.array(sorted(pc1, key=lambda x: (x[0], x[1])))
        pc2_sorted = np.array(sorted(pc2, key=lambda x: (x[0], x[1])))
        # Compare row by row and assert almost equal x, y and z fields
        for i in range(pc1_sorted.shape[0]):
            # There are some floating point errors, if this is fixed, this test
            # should use assertEqual
            self.assertEqual(pc1_sorted[i][0], pc2_sorted[i][0]) # x
            self.assertEqual(pc1_sorted[i][1], pc2_sorted[i][1]) # y
            self.assertEqual(pc1_sorted[i][2], pc2_sorted[i][2]) # z
            self.assertEqual(pc1_sorted[i][3], pc2_sorted[i][3]) # intensity
            self.assertEqual(pc1_sorted[i][4], pc2_sorted[i][4]) # sem_class
            self.assertEqual(pc1_sorted[i][5], pc2_sorted[i][5]) # ins_class


if __name__ == '__main__':
    unittest.main()