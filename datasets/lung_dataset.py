import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TumorSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Dataset for tumor segmentation using Mask R-CNN

        Args:
            root_dir (str): Root directory of the dataset
            transform (callable, optional): Transformations to apply
            mode (str): 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        # Collect all image, annotation, and mask paths
        self.image_paths = []
        self.annotation_paths = []
        self.mask_paths = []

        detection_dir = os.path.join(root_dir, 'detections')
        images_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'masks')

        for folder in os.listdir(detection_dir):
            annotation_folder = os.path.join(detection_dir, folder)
            image_folder = os.path.join(images_dir, folder)
            mask_folder = os.path.join(mask_dir, folder)

            if os.path.isdir(annotation_folder) and os.path.isdir(image_folder):
                for ann_file in os.listdir(annotation_folder):
                    if ann_file.endswith('.txt'):
                        self.annotation_paths.append(os.path.join(annotation_folder, ann_file))
                        image_name = ann_file.replace('.txt', '.png')
                        self.image_paths.append(os.path.join(image_folder, image_name))

                        mask_name = ann_file.replace('.txt', '.png')
                        self.mask_paths.append(os.path.join(mask_folder, mask_name))

        assert len(self.image_paths) == len(self.annotation_paths) == len(self.mask_paths), \
            "Mismatch between images, annotations, and masks!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      # Load image
      img_path = self.image_paths[idx]
      image = Image.open(img_path).convert("RGB")
      image = np.array(image)

      # Load mask
      mask_path = self.mask_paths[idx]
      mask = Image.open(mask_path).convert("L")  # Grayscale mask
      mask = np.array(mask)

      # Load bounding boxes
      ann_path = self.annotation_paths[idx]
      boxes, labels = self.load_annotation(ann_path)

      # Create individual masks for each object
      masks = self.create_individual_masks(mask, boxes)

      # Apply transformations (but NO resizing)
      if self.transform:
          # Only apply non-geometric transforms like normalization, color jitter, etc.
          # Remove any resize transforms from your transform pipeline
          transformed = self.transform(image=image)
          image = transformed['image']

          # Transform masks individually
          if len(masks) > 0:
              transformed_masks = []
              for mask in masks:
                  mask_transformed = self.transform(image=mask)
                  transformed_masks.append(mask_transformed['image'])
              masks = np.array(transformed_masks)

          # NO SCALING OF BOUNDING BOXES - keep them as-is since image size unchanged

      # Convert to tensors
      image = torch.as_tensor(image, dtype=torch.float32)
      if len(image.shape) == 3 and image.shape[2] == 3:  # HWC format
          image = image.permute(2, 0, 1)  # Convert to CHW

      # Normalize image to [0, 1]
      if image.max() > 1:
          image = image / 255.0

      # Handle empty masks case
      if len(masks) == 0:
          masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
          boxes = torch.zeros((0, 4), dtype=torch.float32)
          labels = torch.zeros((0,), dtype=torch.int64)
          area = torch.zeros((0,), dtype=torch.float32)
          iscrowd = torch.zeros((0,), dtype=torch.int64)
      else:
          # Convert masks to tensor
          masks = torch.as_tensor(masks, dtype=torch.uint8)

          # Convert boxes to tensor - NO SCALING NEEDED
          boxes = torch.as_tensor(boxes, dtype=torch.float32)

          # Labels (all tumors have label 1, background is 0)
          labels = torch.ones((len(boxes),), dtype=torch.int64)

          # Area of bounding boxes
          area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

          # All instances are not crowd
          iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

      # Image ID
      image_id = torch.tensor([idx])

      target = {
          "boxes": boxes,
          "labels": labels,
          "masks": masks,
          "image_id": image_id,
          "area": area,
          "iscrowd": iscrowd
      }

      return image, target

    def load_annotation(self, ann_path):
        """Load bounding box annotations"""
        boxes = []
        labels = []

        try:
            with open(ann_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = list(map(int, line.split(",")))
                    if len(data) != 4:
                        print(f"Warning: Invalid annotation format in {ann_path}: {line}")
                        continue

                    x_min, y_min, x_max, y_max = data
                    # Ensure valid bounding box
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(1)  # Tumor class
                    else:
                        print(f"Warning: Invalid bounding box in {ann_path}: {data}")
        except Exception as e:
            print(f"Error reading annotation file {ann_path}: {e}")

        return boxes, labels

    def create_individual_masks(self, mask, boxes):
        """Create individual masks for each bounding box"""
        individual_masks = []

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # Ensure coordinates are within image bounds
            h, w = mask.shape
            x_min = max(0, min(x_min, w-1))
            y_min = max(0, min(y_min, h-1))
            x_max = max(x_min+1, min(x_max, w))
            y_max = max(y_min+1, min(y_max, h))

            # Create a mask for this specific bounding box region
            individual_mask = np.zeros_like(mask)
            roi_mask = mask[y_min:y_max, x_min:x_max]
            individual_mask[y_min:y_max, x_min:x_max] = roi_mask

            # Binarize the mask (assuming mask values > 0 are positive)
            individual_mask = (individual_mask > 0).astype(np.uint8)
            individual_masks.append(individual_mask)

        return np.array(individual_masks) if individual_masks else np.array([])
    

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

