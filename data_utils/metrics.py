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
            iou = intersection / union

        print(f"Class {cls}: Intersection = {intersection}, Union = {union}, IoU = {iou}")

        iou_per_class.append(iou)


    return iou_per_class

