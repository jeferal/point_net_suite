import argparse

from data_utils.dales_dataset import DalesDataset, visualize_pointcloud
from data_utils.s3_dis_dataset import S3DIS
from data_utils.point_cloud_utils import get_point_cloud_limits


if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Visualize DALES dataset')

    parser.add_argument('dataset', type=str, default='dales', help='Dataset to visualize (dales or s3dis)')
    # Argument is the DALES object root path
    parser.add_argument('data_path', type=str, default='data/DALESObjects', help='Root path of the dataset')
    # Second argument is the split
    parser.add_argument('split', type=str, default='train', help='Split of the dataset (train or test)')
    # Third argument is the index of the point cloud
    parser.add_argument('index', type=int, default=0, help='Index of the point cloud')
    # Optional argument to set the list of areas
    parser.add_argument('--areas', type=int, nargs='+', default=[1,2,3,4,5], help='List of areas for s3dis')
    # Optional argument to set the number of partitions
    parser.add_argument('--partitions', type=int, default=5, help='Number of partitions for dales dataset')
    # Optional argument to set the overlap
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap between partitions for dales')
    # 4th argument enables intensity
    parser.add_argument('--intensity', action='store_true', help='Use intensity data for dales')
    # Num points, by default None
    parser.add_argument('--num_points', type=int, default=None, help='Number of points to downsample the point cloud')
    # Rprob
    parser.add_argument('--r_prob', type=float, default=0.25, help='Probability of random rotation')

    # Parse the arguments
    args = parser.parse_args()

    if args.dataset == 's3dis':
        print("Loading S3DIS dataset")
        dataset = S3DIS(args.data_path, area_nums=args.areas, split=args.split, npoints=args.num_points, r_prob=args.r_prob, include_rgb=False)
    elif args.dataset == 'dales':
        print("Loading DALES dataset")
        dataset = DalesDataset(args.data_path, args.split, partitions=args.partitions, intensity=args.intensity, overlap=args.overlap, npoints=args.num_points, normalize=False)
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    print(f"Number of point clouds in the dataset: {len(dataset)}")

    # Allow the user the possibility to navigate over different point clouds
    index = args.index
    while True:
        print(f"Visualizing point cloud {index}")
        point_cloud, labels = dataset[index]

        print(f"Point cloud shape {point_cloud.shape}")
        print(f"Labels shape {labels.shape}")

        # Get the limits of the point cloud
        min_values, max_values = get_point_cloud_limits(point_cloud)

        print(f"X range [{min_values[0]} - {max_values[0]}], diff {max_values[0] - min_values[0]}")
        print(f"Y range [{min_values[1]} - {max_values[1]}], diff {max_values[1] - min_values[1]}")
        print(f"Z range [{min_values[2]} - {max_values[2]}], diff {max_values[2] - min_values[2]}")

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
