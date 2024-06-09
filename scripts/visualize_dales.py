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
    # Optional argument to set the number of partitions
    parser.add_argument('--partitions', type=int, default=5, help='Number of partitions')
    # Optional argument to set the overlap
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap between partitions')
    # 4th argument enables intensity
    parser.add_argument('--intensity', action='store_true', help='Use intensity data')

    # Parse the arguments
    args = parser.parse_args()

    dataset = DalesDataset(args.data_path, args.split, partitions=args.partitions, intensity=args.intensity, overlap=args.overlap)

    # Allow the user the possibility to navigate over different point clouds
    index = args.index
    while True:
        point_cloud, labels = dataset[index]

        print(f"Point cloud shape {point_cloud.shape}")
        print(f"Labels shape {labels.shape}")

        window_name = f"Tile {index} of {len(dataset)}. Press q to exit."

        visualize_pointcloud(point_cloud, labels, window_name)

        # If the user presses right arrow, move to the next point cloud
        # If the user presses left arrow, move to the previous point cloud
        # If the user presses q, exit the program
        # If the user presses space, ask for a new index

        print("Press d arrow to move to the next point cloud")
        print("Press a arrow to move to the previous point cloud")
        print("Enter a number to move to a specific index of tile of the dataset")
        print("Press q to exit the program")

        key = input()

        if key == 'q':
            break
        if key.isdigit():
            index = int(key)
            if index < 0 or index >= len(dataset):
                print("Invalid index")
                continue
        elif key == 'd':
            index = min(index + 1, len(dataset) - 1)
        elif key == 'a':
            index = max(index - 1, 0)
        else:
            print("Invalid key")
            continue
