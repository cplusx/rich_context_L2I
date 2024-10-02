import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class RichContextCOCODataset(Dataset):
    def __init__(
        self,
        root,
        split='train',
        label_dir='',
        image_size=768, # image size for output
        image_size_when_labeling=768, # image size when labeling, the bbox xyxy are scaled to this size

    ):
        image_dir = os.path.join(root, f'{split}2017')
        images = os.listdir(image_dir)
        image_names = [os.path.basename(img).split('.')[0] for img in images]

        valid_image_paths = []
        valid_labels = []

        for image_name in image_names:
            label_path = os.path.join(label_dir, f'label_{image_name}.json')
            if os.path.exists(label_path):
                valid_image_paths.append(os.path.join(image_dir, f'{image_name}.jpg'))
                valid_labels.append(label_path)

        print(f"Found {len(valid_image_paths)} images")
        self.valid_image_paths = valid_image_paths
        self.valid_labels = valid_labels
        self.image_size = image_size
        self.image_size_when_labeling = image_size_when_labeling

        self.set_transforms()

    def __len__(self):
        return len(self.valid_image_paths)

    def transform(self, image, bboxes):
        return self.transform_albumatation(image=image, bboxes=bboxes)

    def set_transforms(self):
        height = width = self.image_size
        self.transform_albumatation = A.Compose([
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1), ratio=(0.95, 1.05), p=1),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='coco'))

    def parse_label(self, label_path):
        with open(label_path, 'r') as f:
            meta = json.load(f)

        caption = meta['caption']
        annos = meta['annos']

        bboxes = [anno['bbox'] for anno in annos]
        category_names = [anno['category_name'] for anno in annos]
        labels = [anno['caption'] for anno in annos]

        return caption, bboxes, category_names, labels
    
    def scale_bounding_box(self, height, width, bboxes):
        if height > width:
            scale = height / self.image_size_when_labeling
        else:
            scale = width / self.image_size_when_labeling
        
        scaled_bboxes = []
        for bbox in bboxes:
            x1, y1, w, h = bbox
            x1, y1, w, h = int(x1 * scale), int(y1 * scale), int(w * scale), int(h * scale)
            x1 = max(0, x1)
            y1 = max(0, y1)
            scaled_bboxes.append([x1, y1, w, h])

        return scaled_bboxes

    def __getitem__(self, index):
        image_path = self.valid_image_paths[index]
        label_path = self.valid_labels[index]

        caption, bboxes, category_names, labels = self.parse_label(label_path)

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        bboxes = self.scale_bounding_box(image.shape[0], image.shape[1], bboxes)

        bboxes_with_labels = []
        for (x1, y1, w, h), label, cate in zip(bboxes, labels, category_names):
            bboxes_with_labels.append([x1, y1, w, h, (label, cate)])

        transformed = self.transform(image, bboxes_with_labels)
        image = transformed['image']
        bboxes_with_labels = transformed['bboxes']

        bboxes, labels, category_names = [], [], []
        for x1, y1, w, h, (label, cate) in bboxes_with_labels:
            bboxes.append([x1, y1, min(x1 + w, self.image_size-1), min(y1 + h, self.image_size-1)])
            labels.append(label)
            category_names.append(cate)

        return {
            'image': image,
            'image_path': image_path,
            'caption': caption,
            'bboxes': np.array(bboxes).astype(np.int16),
            'labels': labels,
            'categories': category_names
        }


class RichContextCOCOEvalDataset(RichContextCOCODataset):
    def transform(self, image, bboxes):
        h, w = image.shape[:2]        
        if h > w:
            # padding to the right side
            pad_size = h - w
            image = np.pad(image, ((0, 0), (0, pad_size), (0, 0)), 'constant', constant_values=255)
            self.pad_size = int(pad_size / h * self.image_size)
            self.pad_side = 'right'
        elif w > h:
            # padding to the bottom side
            pad_size = w - h
            image = np.pad(image, ((0, pad_size), (0, 0), (0, 0)), 'constant', constant_values=255)
            self.pad_size = int(pad_size / w * self.image_size)
            self.pad_side = 'bottom'
        else:
            self.pad_size = 0
            self.pad_side = 'none'
        return self.transform_albumatation(image=image, bboxes=bboxes)

    def set_transforms(self):
        height = width = self.image_size
        self.transform_albumatation = A.Compose([
            A.Resize(height=height, width=width, p=1),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='coco'))

    def __getitem__(self, index):
        res = super().__getitem__(index)
        res['pad_size'] = self.pad_size
        res['pad_side'] = self.pad_side
        return res
