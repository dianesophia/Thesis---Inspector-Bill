from ultralytics import YOLO
import numpy as np
import random

# Define initial hyperparameters
hyperparams = {
    "lr0": 0.01,  # Initial learning rate
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "iou_thres": 0.6
}

# Function to update hyperparameters using gradient-based search
def update_hyperparams(hyperparams, gradients, alpha=0.1):
    for key in hyperparams:
        hyperparams[key] -= alpha * gradients[key]  # Update hyperparameters
        hyperparams[key] = max(0.0001, hyperparams[key])  # Prevent negative values
    return hyperparams

# Train YOLOv8 and get validation loss
def train_yolo(hyperparams):
    model = YOLO("yolov8n.pt")  # Load YOLOv8 nano model
    results = model.train(
        data="coco128.yaml",  # Dataset
        epochs=10,
        imgsz=640,
        lr0=hyperparams["lr0"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"],
        iou_thres=hyperparams["iou_thres"],
        device="cuda"  # Use GPU if available
    )
    return results.results_dict["val/loss"]  # Get validation loss

# AGBS tuning loop
best_loss = float("inf")
best_hyperparams = hyperparams.copy()

for iteration in range(5):  # Tune for 5 iterations
    loss = train_yolo(hyperparams)
    print(f"Iteration {iteration + 1}, Loss: {loss}, Hyperparams: {hyperparams}")

    # Compute gradients (simulated for now)
    gradients = {
        "lr0": -0.01 * loss,
        "momentum": -0.002 * loss,
        "weight_decay": -0.0001 * loss,
        "iou_thres": -0.001 * loss
    }

    # Update hyperparameters using AGBS
    hyperparams = update_hyperparams(hyperparams, gradients)

    # Save best hyperparameters
    if loss < best_loss:
        best_loss = loss
        best_hyperparams = hyperparams.copy()

print("\n🎯 Best Hyperparameters Found:", best_hyperparams)
