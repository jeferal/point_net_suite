import os
import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import torch

# Intersection over union
def compute_iou(targets, predictions):

    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union

def compute_iou_per_class(targets, predictions, num_classes):
    iou_per_class = []

    for cls in range(num_classes):
        cls_target = (targets == cls)
        cls_prediction = (predictions == cls)

        intersection = torch.sum(cls_target & cls_prediction) # true positives
        union = torch.sum(cls_target | cls_prediction).item() # union

        if union == 0:
            iou = float('nan')  # To avoid division by zero
        else:
            # Intersection over union
            iou = intersection / union

        print(f"Class {cls}: Intersection = {intersection}, Union = {union}, IoU = {iou}")

        # Check if iou is nan
        if torch.isnan(iou):
            raise ValueError(f"Class {cls} has an IoU of NaN. This is likely due to the class not being present in the target or prediction tensors.")

        iou_per_class.append(iou)

    return iou_per_class

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def plot_metrics_by_class_grid(train_iou, eval_iou, show=True, save_path=None):
    epochs = range(len(train_iou))
    num_classes = len(train_iou[0])

    # Determine the grid size
    cols = 4
    rows = (num_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharex=True, sharey=True)
    fig.patch.set_facecolor('#2E2E2E')  # Dark background for the figure
    axes = axes.flatten()

    print(f"The number of classes is {num_classes}")
    for class_idx in range(num_classes):
        train_iou_class = [epoch[class_idx] for epoch in train_iou]
        eval_iou_class = [epoch[class_idx] for epoch in eval_iou]

        ax = axes[class_idx]
        ax.plot(epochs, train_iou_class, label=f'Train IoU', color='#1abc9c', linestyle='-', linewidth=2)
        ax.plot(epochs, eval_iou_class, label=f'Eval IoU', color='#3498db', linestyle='--', linewidth=2)

        ax.set_title(f'Class {class_idx}', fontsize=14, color='white')
        ax.set_xlabel('Epochs', fontsize=12, labelpad=0, color='white')
        ax.set_ylabel('IoU', fontsize=12, color='white')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_facecolor('#2E2E2E')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        legend = ax.legend()
        legend.get_frame().set_facecolor('#2E2E2E')  # Set legend background to black
        legend.get_frame().set_edgecolor('white')  # Set legend border to white
        for text in legend.get_texts():
            text.set_color('white')  # Set legend text color to white

    # Hide any unused subplots
    for i in range(num_classes, len(axes)):
        fig.delaxes(axes[i])

    # Add vertical padding between subplots
    plt.subplots_adjust(hspace=4)

    plt.suptitle('IoU over Epochs by Class', fontsize=20, color='white')
    plt.tight_layout(h_pad=3, rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show()

    if save_path:
        print(f"Saving figure in {os.path.abspath(save_path)}")
        plt.savefig(save_path)

def plot_class_distribution(dataset, title, show=True, save_path=None):
    labels = [sample[1] for sample in tqdm.tqdm(dataset)]
    labels = torch.cat(labels).cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title(title)

    if show:
        plt.show()

    if save_path:
        print(f"Saving figure in {os.path.abspath(save_path)}")
        plt.savefig(save_path)
