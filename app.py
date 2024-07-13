import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import time

from models.point_net_v2_sem_segmentation_msg import get_model
from data_utils.dales_dataset import DalesDataset

CATEGORIES = {
    'ground': 0,
    'vegetation': 1,
    'car': 2,
    'truck': 3,
    'powerline': 4,
    'fence': 5,
    'pole': 6,
    'buildings': 7,
}
NUM_CLASSES = len(CATEGORIES)

LABEL_COLORS = {
    0: 'rgb(127, 127, 127)',  # ground: gray
    1: 'rgb(34, 139, 34)',    # vegetation: green
    2: 'rgb(255, 0, 0)',      # car: red
    3: 'rgb(255, 165, 0)',    # truck: orange
    4: 'rgb(0, 0, 255)',      # powerline: blue
    5: 'rgb(255, 255, 0)',    # fence: yellow
    6: 'rgb(75, 0, 130)',     # pole: indigo
    7: 'rgb(0, 191, 255)',    # buildings: deep sky blue
}

@st.cache_resource
def load_model(model_path, num_points, input_dim):
    checkpoint = torch.load(model_path)
    model = get_model(num_points=num_points, m=NUM_CLASSES, input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

@st.cache_data
def load_dataset(data_path, partitions, overlap, intensity, num_points):
    return DalesDataset(
        data_path, 'test', partitions=partitions, overlap=overlap, intensity=intensity, npoints=num_points
    )

def main():
    st.title("Point Cloud Semantic Segmentation")

    model_path = st.text_input("Enter the model path (.pth file):")
    data_path = st.text_input("Enter the data path:")
    num_points = st.number_input("Number of points in each sample:", min_value=1, value=4096)
    partitions = st.number_input("Number of partitions in the dataset:", min_value=1, value=10)
    overlap = st.number_input("Overlap between partitions:", min_value=0.0, value=0.1, step=0.01)

    if model_path and data_path:
        first_layer_shape = torch.load(model_path)['model_state_dict']['classificator.feature_learner.sa1.conv_blocks.0.0.weight'].shape
        input_dim = 4
        model = load_model(model_path, num_points, input_dim)

        dataset = load_dataset(data_path, partitions, overlap, True, num_points)
        dataset_length = len(dataset)
        
        idx = st.number_input("Enter the dataset index to perform inference on:", min_value=0, max_value=dataset_length-1, value=0)
        
        if st.button("Perform Inference"):
            points, _ = dataset[idx]
            st.write(f"Loaded point cloud with shape: {points.shape}")
            
            points = points.unsqueeze(0)
            points = points.transpose(2, 1)

            start_time = time.time()
            pred, _, _ = model(points)
            inference_time = time.time() - start_time

            pred = pred.argmax(dim=2).squeeze().numpy()
            st.write(f"Inference took {inference_time:.2f} seconds")
            # Remove the intensity dimension and create DataFrame with the point cloud and predictions
            points = points.squeeze(0)[:3, :].transpose(1, 0).numpy()  # shape: [4096, 3]
            df = pd.DataFrame(points, columns=['x', 'y', 'z'])
            df['label'] = pred
            df['color'] = df['label'].map(LABEL_COLORS)

            # Visualize the point cloud with predictions
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', title='Point Cloud Segmentation')
            fig.update_traces(marker=dict(size=2))  # Adjust marker size as needed
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()
