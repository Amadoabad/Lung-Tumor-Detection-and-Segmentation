import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from albumentations.pytorch import ToTensorV2


class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, detection_dir, transform=None, task='detection'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.detection_dir = detection_dir
        self.transform = transform
        self.task = task  # 'detection' or 'segmentation'

        self.data = self._gather_data()

    def _gather_data(self):
        data = []
        subjects = os.listdir(self.image_dir)
        for subject in subjects:
            img_folder = os.path.join(self.image_dir, subject)
            mask_folder = os.path.join(self.mask_dir, subject)
            det_folder = os.path.join(self.detection_dir, subject)

            for img_file in os.listdir(img_folder):
                img_path = os.path.join(img_folder, img_file)
                mask_path = os.path.join(mask_folder, img_file)
                txt_file = img_file.rsplit('.', 1)[0] + '.txt'
                txt_path = os.path.join(det_folder, txt_file)

                boxes = []
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        for line in f:
                            coords = tuple(map(int, line.strip().split(',')))
                            boxes.append(coords)

                data.append({
                    'image': img_path,
                    'mask': mask_path,
                    'boxes': boxes
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image = Image.open(record['image']).convert('L')
        image = np.array(image)
        mask = Image.open(record['mask']).convert('L') 
        mask = np.array(mask, dtype=np.float32)
        mask[mask == 255.0] = 1.0
        boxes = record['boxes']

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = ToTensorV2()(image)
            mask = ToTensorV2()(mask)

        if self.task == 'detection':
            target = {'boxes': torch.tensor(boxes, dtype=torch.float32)}
            return image, target
        elif self.task == 'segmentation':
            return image, mask
