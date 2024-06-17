import argparse

from data_utils.metrics import load_metrics, plot_metrics_by_class_grid


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Visualize IoU per class')
    parser.add_argument('file_path', type=str, help='Path to the metrics file')

    args = parser.parse_args()

    metrics = load_metrics(args.file_path)
    train_iou = metrics['train_iou']
    eval_iou = metrics['eval_iou']

    # Plot the metrics by class in a grid
    plot_metrics_by_class_grid(train_iou, eval_iou)
