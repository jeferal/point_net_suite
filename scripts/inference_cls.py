import argparse
import torch
import numpy as np
import time
import open3d as o3d
from models.point_net_classification import get_model
from data_utils.model_net import ModelNetDataLoader

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

    # Get the specified idx from test dataset
    idx = args.index
    points, label = test_dataset[idx]
    tensor = torch.from_numpy(points).unsqueeze(0)
    tensor = tensor.transpose(2, 1)

    # Get the point cloud data from the sample
    start_time = time.time()
    pred, crit_idxs, _ = model(tensor)
    inference_time = time.time() - start_time
    
    ground_truth = test_dataset.cat[label]
    prediction = test_dataset.cat[pred.data.max(1)[1]]
    print(f"Ground truth: {ground_truth}, Prediction: {prediction}. Inference took {inference_time:.2f} seconds")

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()

    if args.show_critical_only:
        # Assign only the critical points
        crit_points = points[crit_idxs.squeeze()]
        point_cloud.points = o3d.utility.Vector3dVector(crit_points)
        crit_colors = np.full((crit_points.shape[0], 3), [1, 0, 0])  # RGB color for red
        point_cloud.colors = o3d.utility.Vector3dVector(crit_colors)
    else:
        # Assign all points
        point_cloud.points = o3d.utility.Vector3dVector(points)
        colors = np.full((points.shape[0], 3), [0, 0, 1])  # RGB color for blue
        colors[crit_idxs.squeeze()] = [1, 0, 0]  # RGB color for red for critical points
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with automatic rotation
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Ground truth: {ground_truth}, Prediction: {prediction}")
    vis.add_geometry(point_cloud)
    
    
    # Rotate the view
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 75.0)  # Set the initial tilt angle to 45 degrees
    for _ in range(360):
        ctr.rotate(30.0, 0.0)  # Rotate by 10 degrees on each iteration
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)  # Pause for 0.1 seconds to slow down the rotation
    
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for classification')
    # First argument is the model path
    parser.add_argument('model_path', type=str, help='Path to the model (.pth file)')
    # Second argument is the data path
    parser.add_argument('data_path', type=str, help='Path to the dataset, for example data/modelnet40')
    # Argument for the number of points, by default 1024
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points in each sample')
    # Argument for showing only critical points
    parser.add_argument('--show_critical_only', action='store_true', help='Show only the critical points')
    # Argument for specifying the index of the test sample
    parser.add_argument('--index', type=int, default=0, help='Index of the test sample to use')

    args = parser.parse_args()

    main(args)
