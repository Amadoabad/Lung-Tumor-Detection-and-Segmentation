import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
from PIL import Image
import numpy as np
import cv2
class MultiObjectSameClassDatasetFixed(Dataset):
    def __init__(self, root_dir, transform=None, target_format="yolo", task="detection"):
        """
        Fixed version of MultiObjectSameClassDataset for DeepLabV3 segmentation.

        Args:
            root_dir (str): Root directory of the dataset (train/test directory).
            transform (callable, optional): Transformation to apply to images.
            target_format (str): Format of targets - "yolo", "faster_rcnn", or "retinanet".
            task (str): Task type - "detection" or "segmentation".
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_format = target_format
        self.task = task  # Either "detection" or "segmentation"

        # Collect all image and annotation paths
        self.image_paths = []
        self.annotation_paths = []
        self.mask_paths = []  # For segmentation task

        detection_dir = os.path.join(root_dir, 'detections')
        images_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'masks')  # Folder for masks (only for segmentation)

        for folder in os.listdir(detection_dir):
            annotation_folder = os.path.join(detection_dir, folder)
            image_folder = os.path.join(images_dir, folder)
            mask_folder = os.path.join(mask_dir, folder)  # Mask folder for segmentation

            if os.path.isdir(annotation_folder) and os.path.isdir(image_folder):
                for ann_file in os.listdir(annotation_folder):
                    if ann_file.endswith('.txt'):
                        self.annotation_paths.append(os.path.join(annotation_folder, ann_file))
                        image_name = ann_file.replace('.txt', '.png')  # Assuming .png format
                        self.image_paths.append(os.path.join(image_folder, image_name))

                        if self.task == "segmentation":
                            # Add corresponding mask path for segmentation
                            mask_name = ann_file.replace('.txt', '.png')  # Assuming the mask format is also .png
                            self.mask_paths.append(os.path.join(mask_folder, mask_name))

        assert len(self.image_paths) == len(self.annotation_paths), "Mismatch between images and annotations!"
        if self.task == "segmentation":
            assert len(self.image_paths) == len(self.mask_paths), "Mismatch between images and masks!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Load mask if the task is segmentation
        if self.task == "segmentation":
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')  # Convert to grayscale
            mask = np.array(mask)

            # Binarize the mask if it's not already binary
            if mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)

            # Apply transform using albumentations if provided
            if self.transform:
                # Make sure transform is albumentations transform for mask compatibility
                if isinstance(self.transform, A.Compose):
                    transformed = self.transform(image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]
                else:
                    # Custom handling for non-albumentations transforms
                    if self.transform:
                        transformed = self.transform(image=image)
                        image = transformed["image"]

                    # Convert mask to tensor manually
                    mask = torch.from_numpy(mask).float()

            # Ensure mask has correct shape for DeepLabV3
            # For DeepLabV3, the mask should be [H, W] or [1, H, W]
            if isinstance(mask, torch.Tensor):
                # If already tensor, just ensure it's 2D or 3D with channel first
                if mask.dim() == 2:  # [H, W]
                    pass  # Already correct shape
                elif mask.dim() == 3 and mask.shape[0] == 1:  # [1, H, W]
                    pass  # Already correct shape
                elif mask.dim() == 3 and mask.shape[0] != 1:  # [H, W, 1] or similar
                    mask = mask.permute(2, 0, 1)  # Convert to [1, H, W]
            else:
                # If numpy array
                if mask.ndim == 2:  # [H, W]
                    mask = torch.from_numpy(mask).float()
                elif mask.ndim == 3 and mask.shape[2] == 1:  # [H, W, 1]
                    mask = torch.from_numpy(mask).float().permute(2, 0, 1)  # Convert to [1, H, W]
                else:
                    mask = torch.from_numpy(mask).float()

            return image, mask

        else:
            # For detection task, follow the original implementation
            height, width = image.shape[:2]  # Retrieve dimensions from NumPy array

            # Load annotation (for detection)
            ann_path = self.annotation_paths[idx]
            boxes, labels = self.load_annotation(ann_path, (width, height))

            # Transform image if needed
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            # Format target based on the requested format
            target = self.format_target(boxes, labels, (width, height))

            return image, target

    def load_annotation(self, ann_path, img_size):
        """
        Load annotation from a file and return the raw bounding box coordinates.
        Now supports multiple bounding boxes per image (all with same class).

        Args:
            ann_path (str): Path to the annotation file.
            img_size (tuple): Size of the image (width, height).

        Returns:
            list: List of bounding boxes in the format [[xmin, ymin, xmax, ymax], ...].
            list: List of labels (all 0, as there's only one class).
        """
        boxes = []
        labels = []

        with open(ann_path, "r") as f:
            for line in f:
                # Read each line as a bounding box
                data = list(map(int, line.strip().split(",")))
                if len(data) != 4:
                    raise ValueError(f"Annotation file {ann_path} has invalid format. Expected 4 values per line.")

                x_min, y_min, x_max, y_max = data
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(0)  # All objects have the same class (0)

        return boxes, labels

    def format_target(self, boxes, labels, img_size):
        """
        Format the target based on the required target format.
        Now handles multiple boxes per image.

        Args:
            boxes (list): List of bounding boxes in [xmin, ymin, xmax, ymax] format.
            labels (list): List of class labels (all same class).
            img_size (tuple): Image size (width, height).

        Returns:
            dict: Formatted target.
        """
        width, height = img_size

        if self.target_format == "yolo":
            # Convert each bounding box to YOLO format: [label, x_center, y_center, width, height]
            formatted_boxes = []
            for box, label in zip(boxes, labels):
                yolo_box = [
                    label,
                    (box[0] + box[2]) / (2 * width),  # x_center (normalized)
                    (box[1] + box[3]) / (2 * height),  # y_center (normalized)
                    (box[2] - box[0]) / width,         # width (normalized)
                    (box[3] - box[1]) / height         # height (normalized)
                ]
                formatted_boxes.append(yolo_box)
            return {"boxes": formatted_boxes, "labels": labels}

        elif self.target_format in ["faster_rcnn", "retinanet"]:
            # Convert to tensors for Faster R-CNN or RetinaNet
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)  # Box coordinates as tensor
            labels_tensor = torch.tensor(labels, dtype=torch.int64)  # Labels as tensor
            return {"boxes": boxes_tensor, "labels": labels_tensor}

# Add a modified version of the MultiObjectSameClassDatasetFixed for box-based cropping
class BoxCroppedSegmentationDataset(MultiObjectSameClassDatasetFixed):
    def __init__(self, root_dir, transform=None, target_format="yolo", max_boxes_per_image=1):
        """
        Modified version of MultiObjectSameClassDatasetFixed that crops images based on bounding boxes.

        Args:
            root_dir (str): Root directory of the dataset (train/test directory).
            transform (callable, optional): Transformation to apply to images.
            target_format (str): Format of targets - "yolo", "faster_rcnn", or "retinanet".
            max_boxes_per_image (int): Maximum number of boxes to crop per image (default: 1)
        """
        super().__init__(root_dir, transform, target_format, task="detection")
        self.max_boxes_per_image = max_boxes_per_image

        # Create a mapping for masks since we need them for segmentation
        self.mask_mapping = {}
        for img_path in self.image_paths:
            img_dir, img_name = os.path.split(img_path)
            parent_dir = os.path.dirname(img_dir)
            mask_dir = os.path.join(parent_dir, "masks", os.path.basename(img_dir))
            mask_path = os.path.join(mask_dir, img_name)
            self.mask_mapping[img_path] = mask_path

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]
        mask_path = self.mask_mapping.get(img_path)

        # Load image using OpenCV for consistency
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Try with PIL if OpenCV fails
                mask = np.array(Image.open(mask_path).convert('L'))
        else:
            # Create an empty mask if no mask file exists
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Binarize the mask if it's not already binary
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        # Get image dimensions
        height, width = image.shape[:2]

        # Load bounding boxes (format: [x_min, y_min, x_max, y_max])
        boxes, _ = self.load_annotation(ann_path, (width, height))

        # If no boxes, return the whole image
        if not boxes:
            # Apply transformations to the whole image
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            return image, mask

        # Select a random box if there are multiple
        if len(boxes) > self.max_boxes_per_image:
            indices = np.random.choice(len(boxes), self.max_boxes_per_image, replace=False)
            selected_boxes = [boxes[i] for i in indices]
        else:
            selected_boxes = boxes[:self.max_boxes_per_image]

        # Choose one box randomly for this iteration
        box_idx = np.random.randint(0, len(selected_boxes))
        box = selected_boxes[box_idx]

        # Extract coordinates
        x_min, y_min, x_max, y_max = box

        # Ensure coordinates are within image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width, x_max), min(height, y_max)

        # Crop image and mask
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=cropped_image, mask=cropped_mask)
            cropped_image = transformed['image']
            cropped_mask = transformed['mask']

        return cropped_image, cropped_mask

# Define transformations
train_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Dataset and DataLoader setup
def create_dataloaders(train_dir, val_dir, batch_size):
    # Create datasets using the BoxCroppedSegmentationDataset
    train_dataset = BoxCroppedSegmentationDataset(
        root_dir=train_dir,
        transform=train_transform,
        target_format="yolo",
        max_boxes_per_image=1  # Use 1 box per image for training
    )

    val_dataset = BoxCroppedSegmentationDataset(
        root_dir=val_dir,
        transform=val_transform,
        target_format="yolo",
        max_boxes_per_image=1  # Use 1 box per image for validation
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
