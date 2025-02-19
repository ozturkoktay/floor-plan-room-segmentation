# Floor Plan Room Semantic Segmentation

## Overview
This project focuses on the semantic segmentation of floor plans using deep learning. The model is trained to classify different components of a floor plan, such as rooms, walls, doors, and windows, using a U-Net architecture with a ResNet encoder. The dataset consists of floor plan images and corresponding segmentation masks, preprocessed and augmented for training.

## Features
- Implements **U-Net** with **ResNet** encoder using **segmentation_models_pytorch**.
- Utilizes **Albumentations** for data augmentation.
- Supports **K-Fold Cross-Validation**.
- Evaluates model performance using **Precision, Recall, F1-Score, Dice Score, and mAP (Mean Average Precision)**.
- Detects room instances with **bounding box extraction and confidence scoring**.
- Provides **visualization tools** for debugging predictions.
- Implements **early stopping** and **learning rate scheduling** for efficient training.

## Installation
### Dependencies
Make sure to install the required dependencies:
```bash
pip install segmentation-models-pytorch albumentations opencv-python numba torch torchvision numpy matplotlib scipy
```

## Dataset Structure
The dataset is expected to be organized as follows:
```
mask-semantic/
│── images/            # Raw input images
│── masks/             # Corresponding segmentation masks
│── train_images/      # Training images
│── train_masks/       # Training masks
│── val_images/        # Validation images
│── val_masks/         # Validation masks
│── test_images/       # Testing images
│── test_masks/        # Testing masks
│── _classes.csv       # Class definitions (Pixel Value, Class Name)
```

## Training Process
### 1. Data Preparation
- Reads the `_classes.csv` file to map pixel values to class indices.
- Loads images and masks into PyTorch **Dataset** and **DataLoader**.
- Applies transformations for data augmentation.

### 2. Model Selection
- Uses `segmentation_models_pytorch` with the **U-Net architecture**.
- Encoder options: EfficientNet, ResNet, PSPNet, DeepLabV3+.
- Model initializes with `num_classes` dynamically derived from the dataset.

### 3. Training
- Uses **CrossEntropyLoss** for multi-class segmentation.
- Optimizer: **Adam** with learning rate scheduling.
- Metrics: **Dice coefficient, Precision, Recall, F1-Score, and Accuracy**.
- Implements **early stopping** based on validation loss.

### 4. K-Fold Cross-Validation
- Performs **5-Fold cross-validation** to evaluate model generalization.
- Saves the best model based on validation performance.

## Evaluation
- Calculates **Precision, Recall, F1-Score, and Confusion Matrix**.
- Implements **mAP (Mean Average Precision) from IoU thresholds**.
- Visualizes **incorrect predictions** for analysis.

## Model Inference & Room Detection
- Loads trained model weights from a `.pth` file.
- Identifies connected components for `room` class.
- Generates **bounding boxes and confidence scores** for detected rooms.
- Outputs results in **JSON format**.

## Example Output
```json
{
  "predictions": [
    {
      "x": 320.5,
      "y": 240.2,
      "width": 150.0,
      "height": 120.0,
      "confidence": 0.85,
      "class": "room",
      "class_id": 3,
      "detection_id": "abc123-uuid"
    }
  ]
}
```

## Visualization Tools
- Displays **input images, ground truth masks, and predicted masks**.
- Highlights **room instances with bounding boxes**.
- Applies **morphological erosion** to refine predictions.
- Supports **interactive visualization** using Matplotlib.

## Saving & Loading the Model
```python
# Save the trained model
torch.save(model.state_dict(), "unet_floorplan_multiclass.pth")

# Load the trained model
model.load_state_dict(torch.load("unet_floorplan_multiclass.pth"))
model.eval()
```

## Results & Performance
- The model achieves high **Dice scores** and **mAP** for room segmentation.
- Visualization of incorrect predictions helps improve model performance.

## Future Improvements
- Train on a larger dataset for better generalization.
- Experiment with **self-supervised learning** for pretraining.
- Implement **active learning** to improve annotation efficiency.

## License
MIT License