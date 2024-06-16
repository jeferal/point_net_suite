import json
import matplotlib.pyplot as plt

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def plot_metrics(train_iou, eval_iou):
    # Check lengths of arrays
    print(f"Length of train_iou: {len(train_iou)}")
    print(f"Length of eval_iou: {len(eval_iou)}")
    
    epochs = min(len(train_iou), len(eval_iou))

    plt.figure(figsize=(12, 6))
    
    
    # Set background color
    plt.gca().set_facecolor('#2E2E2E')

    # Plot Train IoU
    plt.plot(range(epochs), train_iou[:epochs], label='Train IoU', color='#1abc9c', linewidth=2.5)

    # Plot Eval IoU
    plt.plot(range(epochs), eval_iou[:epochs], label='Eval IoU', color='#3498db', linewidth=2.5)

    # Adding titles and labels with a white color
    plt.xlabel('Epochs', fontsize=14, color='black')
    plt.ylabel('IoU', fontsize=14, color='black')
    plt.title('IoU over Epochs', fontsize=16, color='black')

    # Adding grid lines
    plt.grid(color='#707070', linestyle='--', linewidth=0.7)

    # Adding legend with white text
    legend = plt.legend(loc='best', fontsize=12)
    for text in legend.get_texts():
        text.set_color('white')

    # Customizing tick parameters
    plt.tick_params(colors='black', which='both')
    plt.suptitle('Training and Evaluation IoU Metrics', fontsize=20, color='black')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the metrics
    metrics = load_metrics(r'C:\Users\pizza\PycharmProjects\point_net_suite\iou_metrics.json')
    train_iou = metrics['train_iou']
    eval_iou = metrics['eval_iou']

    # Plot the metrics
    plot_metrics(train_iou, eval_iou)
