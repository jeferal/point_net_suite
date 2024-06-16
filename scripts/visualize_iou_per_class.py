import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def plot_metrics_by_class_grid(train_iou, eval_iou):
    epochs = range(len(train_iou))
    num_classes = len(train_iou[0])
    
    # Determine the grid size
    cols = 4
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharex=True, sharey=True)
    fig.patch.set_facecolor('#2E2E2E')  # Dark background for the figure
    axes = axes.flatten()

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
    plt.show()


if __name__ == "__main__":
    # Load the metrics
    file_path = r'C:\Users\pizza\PycharmProjects\point_net_suite\iou_metrics.json'
    metrics = load_metrics(file_path)
    train_iou = metrics['train_iou']
    eval_iou = metrics['eval_iou']

    # Plot the metrics by class in a grid
    plot_metrics_by_class_grid(train_iou, eval_iou)
