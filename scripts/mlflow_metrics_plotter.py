import os
import argparse

import mlflow
import matplotlib.pyplot as plt
from data_utils.metrics import plot_metrics_by_class_grid

def get_metric(client : mlflow.client, run_id : str, key : str) -> list:
    """
        This method reads a metric and return a list with the values
        of length, which is all steps.
    """

    # Get the metric
    metric = client.get_metric_history(run_id, key)

    # Get the values
    values = [m.value for m in metric]

    return values

def get_metrics_for_epochs(client: mlflow.client, run_id: str, metric_names: list) -> list:
    """
    This function retrieves metrics for all epochs for each metric name.
    Returns a list where each element is a sublist containing values for each class per epoch.
    """
    epochs = []
    
    # Iterate over each metric name
    for metric_name in metric_names:
        # Get values for the current metric name
        values = get_metric(client, run_id, metric_name)
        # Append values to epochs list
        epochs.append(values)
    
    # Transpose epochs list to get the desired structure
    epochs_per_class_per_epoch = list(map(list, zip(*epochs)))
    
    return epochs_per_class_per_epoch

def main(args):
    # Set the tracking URI to the server
    mlflow.set_tracking_uri(args.url)

    # Get the run ID
    run_id = args.run_id

    # Create a folder with the run id name to store the plots
    if not os.path.exists(run_id):
        os.makedirs(run_id)

    # Metrics name
    train_iou_per_class_metric_name = [
        "train_iou_class_0",
        "train_iou_class_1",
        "train_iou_class_2",
        "train_iou_class_3",
        "train_iou_class_4",
        "train_iou_class_5",
        "train_iou_class_6",
        "train_iou_class_7",
    ]

    eval_iou_per_class_metric_name = [
        "eval_iou_class_0",
        "eval_iou_class_1",
        "eval_iou_class_2",
        "eval_iou_class_3",
        "eval_iou_class_4",
        "eval_iou_class_5",
        "eval_iou_class_6",
        "eval_iou_class_7",
    ]

    # Get the metrics for the specified run
    client = mlflow.tracking.MlflowClient()

    # Create a plot that shows in the same grapth the train loss and eval loss
    # Get the train loss
    train_loss = get_metric(client, run_id, "train_loss")
    eval_loss = get_metric(client, run_id, "eval_loss")

    plt.figure(figsize=(15, 5))
    plt.title("Train vs Eval Loss")
    plt.plot(train_loss, label="Train Loss")
    plt.plot(eval_loss, label="Eval Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(run_id, f"{run_id}_loss.png"))

    # Create a plot that shows the train accuracy and eval accuracy
    train_accuracy = get_metric(client, run_id, "train_accuracy")
    eval_accuracy = get_metric(client, run_id, "eval_accuracy")

    plt.figure(figsize=(15, 5))
    plt.title("Train vs Eval Accuracy")
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(eval_accuracy, label="Eval Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(run_id, f"{run_id}_accuracy.png"))

    # Get the per class iou metrics
    train_iou = get_metrics_for_epochs(client, run_id, train_iou_per_class_metric_name)
    eval_iou = get_metrics_for_epochs(client, run_id, eval_iou_per_class_metric_name)

    # Plot the per class iou metrics
    save_path = os.path.join(run_id, f"{run_id}_iou.png")
    # Test if train_iou and eval_iou have the same length, if not, remove the last elements from the longest one
    # until they have the same length
    while len(train_iou) > len(eval_iou):
        train_iou.pop()
    while len(eval_iou) > len(train_iou):
        eval_iou.pop()
    plot_metrics_by_class_grid(train_iou, eval_iou, show=False, save_path=save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot metrics from MLFlow")

    # Argument that is the mlflow url, by default http://34.77.117.226:9090
    parser.add_argument("--url", type=str, default="http://34.77.117.226:9090")
    # Argument that is the run id
    parser.add_argument("--run_id", type=str, required=True)

    args = parser.parse_args()

    main(args)
