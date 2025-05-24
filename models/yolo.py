
import os
import torch
import numpy as np
import yaml
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import shutil
from datasets_class.lung_dataset import LungDataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def format_target_to_yolo(boxes, img_size):
    """
    Format the target to YOLO format.
    Handles multiple boxes per image and empty boxes (no objects).

    Args:
        boxes (list): List of bounding boxes in [xmin, ymin, xmax, ymax] format.
        img_size (tuple): Image size (width, height).

    Returns:
        list: List of YOLO format annotations (empty if no objects).
    """
    width, height = img_size
    yolo_annotations = []

    # If no boxes, return empty list (no objects in image)
    if not boxes:
        return yolo_annotations

    for box in boxes:
        # Convert to YOLO format: [label, x_center, y_center, width, height]
        x_center = (box[0] + box[2]) / (2 * width)  # x_center (normalized)
        y_center = (box[1] + box[3]) / (2 * height)  # y_center (normalized)
        box_width = (box[2] - box[0]) / width        # width (normalized)
        box_height = (box[3] - box[1]) / height      # height (normalized)

        # Using class label 1 for objects (change if you have different classes)
        yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    return yolo_annotations




def prepare_yolo_dataset(dataset, output_dir):
    """
    Prepare YOLOv8 compatible dataset structure from our custom dataset
    Now includes subject name in filenames to handle duplicate image names across subjects
    """
    # Create directory structure
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    print(len(dataset.data))
    # Process each image and its annotations
    for i in range(len(dataset.data)):
        img_path = dataset.data[i]["image"]
        boxes = dataset.data[i]["boxes"]

        # Get subject name (assuming it's in the path)
        # This extracts the parent directory name as the subject name
        subject_name = os.path.basename(os.path.dirname(img_path))

        # Get image name without path
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]

        # Create new filename with subject prefix
        new_img_name = f"{subject_name}_{img_name}"
        new_img_name_no_ext = f"{subject_name}_{img_name_no_ext}"

        # Load image to get dimensions
        image = Image.open(img_path)
        width, height = image.size

        # Load and convert annotations
        yolo_annotations = format_target_to_yolo(boxes, (width, height))

        # Copy image to new location with new name
        dest_img_path = os.path.join(images_dir, new_img_name)
        shutil.copy(img_path, dest_img_path)

        # Save YOLO annotations with new name
        label_filename = f"{new_img_name_no_ext}.txt"
        with open(os.path.join(labels_dir, label_filename), 'w') as f:
            for ann in yolo_annotations:
                f.write(f"{ann}\n")

    return output_dir

def create_yolo_data_yaml(train_dir, val_dir, output_path="dataset.yaml"):
    """
    Create a YAML file for YOLOv8 training
    """
    data = {
        'path': os.path.abspath('.'),
        'train': os.path.abspath(os.path.join(train_dir, 'images')),
        'val': os.path.abspath(os.path.join(val_dir, 'images')),
        'names': {0: 'object'},  # Single class as per the dataset
        'nc': 1  # Number of classes
    }

    with open(output_path, 'w') as f:
        yaml.dump(data, f)

    return output_path


def load_or_create_model(model_path=None, config_path=None):
    """Load existing model or create new one"""
    try:
        model = YOLO(model_path)
        print(f"Loaded existing model from {model_path}")
        return model
    except:
        print(f"Creating new model{' from ' + config_path if config_path else ''}")
        model = YOLO(config_path if config_path else 'yolov8n.yaml')
        model.save(model_path)
        return model




train_dataset = LungDataset(
    image_dir="Dataset/train/images",
    mask_dir="Dataset/train/masks",
    detection_dir = "Dataset/train/detections"
)

val_dataset = LungDataset(
    image_dir="Dataset/val/images",
    mask_dir="Dataset/val/masks",
    detection_dir = "Dataset/val/detections"
)


# Create YOLO format datasets
yolo_train_dir = "yolo_train"
yolo_val_dir = "yolo_val"

print("Preparing training dataset...")
prepare_yolo_dataset(train_dataset, yolo_train_dir)
print("Preparing validation dataset...")
prepare_yolo_dataset(val_dataset, yolo_val_dir)

# Create YAML config file
yaml_path = create_yolo_data_yaml(yolo_train_dir, yolo_val_dir)
print(f"Created YAML config at: {yaml_path}")


# Initialize YOLOv8 model without pretrained weights
#model = YOLO('yolov8n.yaml')  # Create a new model from scratch
# yolo_model = YOLO(shehapkhalil_yamlfile_path+'/yolo.yaml')

