import argparse

import random
import torch
import numpy as np
import time

import open3d as o3d

from point_net_suite.models.pointnet_sem_segmentation import get_model
from point_net_suite.data_utils.s3_dis_dataset import S3DIS
from torch.utils.data import DataLoader


CATEGORIES = {
    'ceiling'  : 0, 
    'floor'     : 1, 
    'wall'     : 2, 
    'beam'     : 3, 
    'column'   : 4, 
    'window'   : 5,
    'door'     : 6, 
    'table'    : 7, 
    'chair'    : 8, 
    'sofa'     : 9, 
    'bookcase' : 10, 
    'board'    : 11,
    'stairs'   : 12,
    'clutter'  : 13
}
NUM_CLASSES = len(CATEGORIES)

def main(args):
    model = get_model(num_points=4096, m=NUM_CLASSES)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    root_data_path = args.data_path

    loader_args = type('', (), {})()
    loader_args.num_points = 4096
    args.test_area = [3]
    test_dataset = S3DIS(root=root_data_path, area_nums=args.test_area, split='test', npoints=loader_args.num_points)

    # Get a random idx from test dataset
    idx = random.randint(0, len(test_dataset))
    points, label = test_dataset[idx]
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
    pred, crit_idxs, _ = model(input)
    inference_time = time.time() - start_time
    
    # Pred has the shape [1, point_number, num_classes]
    # We want to get the class with the highest probability for each point
    pred = pred.argmax(dim=2)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Create a color map for the classes
    colors = np.zeros((points.shape[0], 3))
    for i in range(NUM_CLASSES):
        colors[pred[0] == i] = np.array([random.random(), random.random(), random.random()])
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud], window_name=f"Inference took {inference_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for semantic segmentation')
    # First argument is the model path
    parser.add_argument('model_path', type=str, help='Path to the model ().pth file)')
    # Second argument is the data path
    parser.add_argument('data_path', type=str, help='Path to the dataset, for example data/stanford_indoor3d')

    args = parser.parse_args()

    main(args)
