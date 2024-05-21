import argparse

import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read .OFF file and extract vertices
def read_off(file_path):
  mesh = trimesh.load_mesh(file_path)
  vertices = mesh.vertices
  return vertices

# Function to visualize point cloud
def visualize_point_cloud(vertices):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  # Extract x, y, z coordinates
  x = vertices[:, 0]
  y = vertices[:, 1]
  z = vertices[:, 2]
  
  # Scatter plot of points
  ax.scatter(x, y, z, c='r', marker='o')
  
  # Set plot labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  plt.show()

# Main function
def main():

  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path', type=str, help='Path to .OFF file')
  args = parser.parse_args()

  vertices = read_off(args.file_path)
  visualize_point_cloud(vertices)

if __name__ == '__main__':
    main()
