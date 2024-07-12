import argparse

import random
import torch
import numpy as np
import time

import open3d as o3d

from models.point_net_classification import get_model
from data_utils.model_net import ModelNetDataLoader
#from torch.utils.data import DataLoader


def main(args):
    model = get_model(num_points=args.num_points, k=40)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    root_data_path = args.data_path

    test_dataset = ModelNetDataLoader(
        root=root_data_path,
        split='test',
        pre_process_data=False,
        num_point=args.num_points)

    # Get a random idx from test dataset
    idx = random.randint(0, len(test_dataset))
    points, label = test_dataset[idx]
    tensor = torch.from_numpy(points).unsqueeze(0)
    tensor = tensor.transpose(2, 1)

    # Get the point cloud data from the sample
    # The critical idxs is a tensor that contains the indices of the points that are critical for the classification
    # The model is doing an inference here
    start_time = time.time()
    pred, crit_idxs, _ = model(tensor)
    inference_time = time.time() - start_time
    
    ground_truth = test_dataset.cat[label]
    prediction = test_dataset.cat[pred.data.max(1)[1]]
    print(f"Ground truth: {ground_truth}, Prediction: {prediction}. Inference took {inference_time:.2f} seconds")

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    # Assign the numpy array to the Open3D point cloud object
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Set specific points to red
    # Initialize all points to a default color (e.g., blue)
    colors = np.full((points.shape[0], 3), [0, 0, 1])  # RGB color for blue
    # Set specific points to red
    for idx in crit_idxs:
        colors[idx] = [1, 0, 0]  # RGB color for red
    # Assign the colors to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # Visualize the point cloud
    # Add a title to the window
    o3d.visualization.draw_geometries([point_cloud], window_name=f"Ground truth: {ground_truth}, Prediction: {prediction}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for classification')
    # First argument is the model path
    parser.add_argument('model_path', type=str, help='Path to the model ().pth file)')
    # Second argument is the data path
    parser.add_argument('data_path', type=str, help='Path to the dataset, for example data/modelnet40')
    # Argument that is the number of points, by default 1024
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points in each sample')

    args = parser.parse_args()

    main(args)
