import argparse
import open3d as o3d

from data_utils.dales_dataset import DalesDataset, visualize_pointcloud


if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Visualize DALES dataset')
    # First argument is the DALES object root path
    parser.add_argument('data_path', type=str, default='/home/jesusferrandiz/Learning/point_net_ws/src/point_net_suite/data/DALESObjects', help='Root path of the DALES objects')
    # Second argument is the split
    parser.add_argument('split', type=str, default='train', help='Split of the dataset (train or test)')
    # Third argument is the index of the point cloud
    parser.add_argument('index', type=int, default=0, help='Index of the point cloud')
    # 4th argument enables intensity
    parser.add_argument('--intensity', action='store_true', help='Use intensity data')

    # Parse the arguments
    args = parser.parse_args()

    dataset = DalesDataset(args.data_path, args.split, intensity=args.intensity)

    # Get the length of the dataset
    print(f"Length of the dataset: {len(dataset)}")
    # Access to a point cloud
    point_cloud, labels = dataset[args.index]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Labels shape: {labels.shape}")

    visualize_pointcloud(point_cloud, labels)
