import argparse

import open3d as o3d

from data_utils.dales_dataset import DalesDataset

if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Visualize DALES dataset')
    # First argument is the DALES object root path
    parser.add_argument('data_path', type=str, default='/home/jesusferrandiz/Learning/point_net_ws/src/point_net_suite/data/DALESObjects', help='Root path of the DALES objects')
    # Second argument is the split
    parser.add_argument('split', type=str, default='train', help='Split of the dataset (train or test)')
    # Third argument is the index of the point cloud
    parser.add_argument('index', type=int, default=0, help='Index of the point cloud')
    # Optional argument to set the number of partitions
    parser.add_argument('--partitions', type=int, default=5, help='Number of partitions')
    # Optional argument to set the overlap
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap between partitions')
    # Optional argument to set the intensity
    parser.add_argument('--intensity', action='store_true', help='Include intensity in the point cloud')

    # Parse the arguments
    args = parser.parse_args()

    dataset = DalesDataset(args.data_path, args.split, partitions=args.partitions, intensity=args.intensity, overlap=args.overlap)

    # Get the length of the dataset
    print(f"Length of the dataset: {len(dataset)}")
    # Access to a point cloud
    point_cloud, labels = dataset[args.index]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Labels shape: {labels.shape}")

    # Convert the point cloud to numpy
    points = point_cloud.numpy()

    # Remove the intensity
    points = points[:, :3]

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    # Assign the numpy array to the Open3D point cloud object
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud], window_name=f"DALES point cloud {args.index}")
