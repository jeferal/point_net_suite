# Import necessary libraries
import numpy as np
import open3d as o3d
from plyfile import PlyData
import matplotlib.pyplot as plt

# Function to explore a PLY file and visualize the point cloud
def explore_and_visualize_ply(file_path):
    # Read the PLY file
    ply_data = PlyData.read(file_path)
    
    # Print the elements and properties in the file
    print(f"Exploring PLY file: {file_path}")
    for element in ply_data.elements:
        print(f"Element: {element.name}")
        print("Properties:")
        for prop in element.properties:
            print(f"  - {prop.name}")

    # Identify the main element (usually 'vertex' or similar)
    main_element_name = ply_data.elements[0].name
    vertex_data = ply_data[main_element_name]
    
    # Print the properties of the main element
    for prop in vertex_data.data.dtype.names:
        print(f"Property: {prop}, dtype: {vertex_data.data[prop].dtype}, shape: {vertex_data.data[prop].shape}")

    # Extract the data
    x = vertex_data.data['x']
    y = vertex_data.data['y']
    z = vertex_data.data['z']
    intensity = vertex_data.data['intensity']
    sem_class = vertex_data.data['sem_class']
    ins_class = vertex_data.data['ins_class']

    # Print some basic statistics for coordinates
    print("\nStatistics for coordinates:")
    print(f"  X: min={x.min()}, max={x.max()}, mean={x.mean()}")
    print(f"  Y: min={y.min()}, max={y.max()}, mean={y.mean()}")
    print(f"  Z: min={z.min()}, max={z.max()}, mean={z.mean()}")

    # Print the first few points for inspection
    print("\nFirst few points:")
    for i in range(min(10, len(vertex_data.data))):
        print(vertex_data.data[i])

    # Prepare data for visualization
    points = np.vstack((x, y, z)).T
    sem_class = np.array(sem_class)

    # Example color palette for semantic classes
    color_palette = np.array([
        [1, 3, 117],    # Class 0: Unknown
        [1, 79, 156],   # Class 1: Blue: Ground
        [1, 114, 3],    # Class 2: Green: Vegetation
        [222, 47, 225], # Class 3: Pink: Cars
        [237, 236, 5],  # Class 4: Yellow: Trucks
        [2, 205, 1],    # Class 5: Light Green: Power Lines
        [5, 216, 223],  # Class 6: Light Blue: Fences
        [250, 125, 0],  # Class 7: Orange: Poles
        [196 ,1, 1],    # Class 8: Red: Buildings
    ])

    # Normalize the palette to [0, 1] range for Open3D
    color_palette = color_palette / 255.0

    # Get the colors based on semantic class
    colors = color_palette[sem_class % len(color_palette)]

    # Normalize intensity to the range [0, 1]
    intensity = intensity.astype(np.float32)
    intensity_min = intensity.min()
    intensity_max = intensity.max()
    intensity = (intensity - intensity_min) / (intensity_max - intensity_min)

    # Calculate normals for the point cloud
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Define a light source direction
    light_direction = np.array([1.0, 1.0, 1.0])
    light_direction /= np.linalg.norm(light_direction)  # Normalize the light direction

    # Get the normals
    normals = np.asarray(o3d_point_cloud.normals)

    # Compute the shading (dot product between normals and light direction)
    shading = np.dot(normals, light_direction)
    shading = np.clip(shading, 0, 1)  # Ensure shading is within [0, 1]

    # Blend intensity with colors, giving more weight to colors
    intensity_weight = 0.1  # Adjust this to give less weight to intensity
    blended_colors = colors * (1 - intensity_weight) + colors * intensity[:, np.newaxis] * intensity_weight

    # Apply the shading effect
    shadow_effect = 0.4  # Adjust this to control shadow intensity
    blended_colors = blended_colors * ((1 - shadow_effect) + shading[:, np.newaxis] * shadow_effect)

    # Ensure colors are within [0, 1]
    blended_colors = np.clip(blended_colors, 0, 1)

    # Create an Open3D point cloud object
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d_point_cloud.colors = o3d.utility.Vector3dVector(blended_colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([o3d_point_cloud], window_name=f"DALES point cloud")

# Path to the PLY file (replace with your own path)
file_path = "C:/Users/pizza/PycharmProjects/point_net_suite/data/DALESObjects/test/5135_54435_new.ply"

# Explore and visualize the PLY file
explore_and_visualize_ply(file_path)
