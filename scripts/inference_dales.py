import argparse

import random
import torch
import numpy as np
import time

import open3d as o3d

from models.point_net_v2_sem_segmentation_msg import get_model

from data_utils.dales_dataset import DalesDataset, visualize_pointcloud


CATEGORIES = {
    'ground'         : 0, 
    'vegetation'     : 1, 
    'car'            : 2, 
    'truck'          : 3, 
    'powerline'      : 4, 
    'fence'          : 5,
    'pole'           : 6, 
    'buildings'      : 7, 
}
NUM_CLASSES = len(CATEGORIES)

def main(args):
    # Get the shape of the first layer of the model
    # to know if the architecture accepts extra features (intensity) or not
    checkpoint = torch.load(args.model_path)
    first_layer_shape = checkpoint['model_state_dict']['classificator.feature_learner.sa1.conv_blocks.0.0.weight'].shape
    input_dim = first_layer_shape[1]

    model = get_model(num_points=args.num_points, m=NUM_CLASSES, input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    print("Loaded model")

    intensity = False
    if input_dim == 4:
        intensity = True

    root_data_path = args.data_path
    test_dataset = DalesDataset(
        root_data_path, 'test',
        partitions=args.partitions, overlap=args.overlap,
        intensity=intensity, npoints=args.num_points)

    # Get a random idx from test dataset
    idx = 69
    points, _ = test_dataset[idx]
    print(points.shape)
    """
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    # Assign the numpy array to the Open3D point cloud object
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])
    """
    input = points.unsqueeze(0)
    input = input.transpose(2,1)

    # Get the point cloud data from the sample
    # The critical idxs is a tensor that contains the indices of the points that are critical for the classification
    # The model is doing an inference here
    start_time = time.time()
    print(f"Intput shape: {input.shape}")
    pred, _, _ = model(input)
    inference_time = time.time() - start_time

    # Pred has the shape [1, point_number, num_classes]
    # We want to get the class with the highest probability for each point
    pred = pred.argmax(dim=2)

    window_name=f"Inference took {inference_time:.2f} seconds"
    visualize_pointcloud(points, pred.squeeze(), window_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for semantic segmentation')
    # First argument is the model path
    parser.add_argument('model_path', type=str, help='Path to the model ().pth file)')
    # Second argument is the data path
    parser.add_argument('data_path', type=str, help='Path to the dataset, for example data/stanford_indoor3d')
    # Another argument that is the number of points, by default 4096
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points in each sample')
    # Argument that is the number of partitions, by default 10
    parser.add_argument('--partitions', type=int, default=10, help='Number of partitions in the dataset')
    # Argument that is the overlap, by default 0.1
    parser.add_argument('--overlap', type=float, default=0.1, help='Overlap between partitions')

    args = parser.parse_args()

    main(args)
