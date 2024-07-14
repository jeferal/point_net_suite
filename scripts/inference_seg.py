import argparse
import random
import torch
import numpy as np
import time
import open3d as o3d

from models.point_net_sem_segmentation import get_model
from data_utils.s3_dis_dataset import S3DIS
# from torch.utils.data import DataLoader

CATEGORIES = {
    'ceiling': 0,
    'floor': 1,
    'wall': 2,
    'beam': 3,
    'column': 4,
    'window': 5,
    'door': 6,
    'table': 7,
    'chair': 8,
    'sofa': 9,
    'bookcase': 10,
    'board': 11,
    'stairs': 12,
    'clutter': 13
}
NUM_CLASSES = len(CATEGORIES)

def main(args):
    model = get_model(num_points=args.num_points, m=NUM_CLASSES)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    root_data_path = args.data_path

    loader_args = type('', (), {})()
    loader_args.num_points = args.num_points
    args.test_area = [3]
    test_dataset = S3DIS(root=root_data_path, area_nums=args.test_area, split='test', npoints=loader_args.num_points)
    
    # Determine the index to use
    if args.idx is None:
        idx = random.randint(0, len(test_dataset) - 1) # Ensure idx is within the valid range
    else:
        idx = args.idx

    points, label = test_dataset[idx]
    print(points.shape)

    input = torch.tensor(points).unsqueeze(0)
    input = input.transpose(2, 1)

    start_time = time.time()
    print(f"Input shape: {input.shape}")
    pred, crit_idxs, _ = model(input)
    inference_time = time.time() - start_time

    pred = pred.argmax(dim=2)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Create a color map for the classes
    colors = np.zeros((points.shape[0], 3))
    
    for i in range(NUM_CLASSES):
        colors[pred[0] == i] = np.array([random.random(), random.random(), random.random()])
        #crit_colors = np.array([1, 0, 0])  # Red color for critical points
        #colors[crit_idxs[0]] = crit_colors
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Inference took {inference_time:.2f} seconds")
    vis.add_geometry(point_cloud)

    # Rotate the view
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 45.0)  # Set the initial tilt angle
    for _ in range(360):
        ctr.rotate(20.0, 0.0, 20.0)  # Rotate by 30 degrees on each iteration
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)  # Pause for 0.1 seconds to slow down the rotation

    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for semantic segmentation')
    parser.add_argument('model_path', type=str, help='Path to the model (.pth file)')
    parser.add_argument('data_path', type=str, help='Path to the dataset, for example data/stanford_indoor3d')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points in each sample')
    parser.add_argument('--idx', type=int, default=None, help='Index of the sample to visualize (default: random)')
    parser.add_argument('--show_critical_only', action='store_true', help='Only show critical points')
    args = parser.parse_args()

    main(args)
