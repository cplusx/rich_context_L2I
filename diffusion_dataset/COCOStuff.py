'''
modified from https://github.com/wtliao/layout2img/blob/c915f15a571d672973ec5722a7711988b70d2bb2/data/cocostuff_loader_ours.py
'''
import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .dataset_utils import bounding_box_to_mask

def get_coco_id_mapping(
    instance_path="data/instances_val2017.json",
    stuff_path="data/stuff_val2017.json", 
    subset_index=None
):
    import json
    def load_one_file(file_path):
        with open(file_path, 'r') as IN:
            data = json.load(IN)
        id_mapping = {}
        for item in data['categories']:
            item_id = item['id']
            item_name = item['name']
            id_mapping[item_id] = item_name
        if subset_index is not None:
            id_mapping = {id_mapping[i] for i in subset_index}
        return id_mapping
    instance_mapping = load_one_file(instance_path)
    stuff_mapping = load_one_file(stuff_path)

    instance_mapping.update(stuff_mapping)
    return instance_mapping


def get_cocostuff_dataset(root="./data", image_size=256, max_objects_per_image=100):
    train_set = CocoStuffBboxDataset(
        root=root,
        image_size=image_size,
        stuff_only=True,
        max_objects_per_image=max_objects_per_image,
        validation=False
    )
    val_set = CocoStuffBboxDataset(
        root=root,
        image_size=image_size,
        stuff_only=True,
        max_objects_per_image=8, # for testing, we use the same config as previous works
        validation=True
    )
    return train_set, val_set

def get_cocostuff_caption_dataset(
    root="./data", 
    image_size=256, 
    max_objects_per_image=100,
    train_empty_string=0,
    val_empty_string=0
):
    train_set = CocoStuffBboxCaptionDataset(
        root = root,
        image_size=image_size,
        stuff_only=True,
        max_objects_per_image=max_objects_per_image,
        empty_string=train_empty_string,
        validation=False
    )
    val_set = CocoStuffBboxCaptionDataset(
        root = root,
        image_size=image_size,
        stuff_only=True,
        max_objects_per_image=8, # for testing, we use the same config as previous works
        empty_string=val_empty_string,
        validation=True
    )
    return train_set, val_set


class CocoStuffBboxDataset(Dataset):
    def __init__(self, root,
                 stuff_only=True, image_size=512, 
                 max_samples=None,
                 min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None,
                 validation=False):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        self.root = root
        image_dir = os.path.join(root, 'train2017') if not validation else os.path.join(root, 'val2017')
        instances_json = os.path.join(root, 'annotations', 'instances_train2017.json') if not validation else os.path.join(root, 'annotations', 'instances_val2017.json')
        stuff_json = os.path.join(root, 'annotations', 'stuff_train2017.json') if not validation else os.path.join(root, 'annotations', 'stuff_val2017.json')

        self.validation = validation
        if stuff_only and stuff_json is None:
            print('WARNING: Got inst_imstuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir = image_dir
        self.max_samples = max_samples
        self.max_objects_per_image = max_objects_per_image
        self.left_right_flip = left_right_flip
        self.set_image_size(image_size)

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            # box_area = object_data['area'] / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                # box_area = object_data['area'] / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                    self.image_id_to_objects[image_id].append(object_data)

            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

    def set_image_size(self, image_size):
        height = width = image_size
        self.transform = A.Compose([
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.1), ratio=(0.95, 1.05), p=1),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='coco', min_visibility=0.1))
        self.image_size = image_size

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            if self.left_right_flip:
                return len(self.image_ids) * 2
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def get_data_of_index(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        flip = False
        if index >= len(self.image_ids):
            index = index - len(self.image_ids)
            flip = True
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size

                objs, boxes, masks = [], [], []  # for discriminator
                objs_f, boxes_f, objs_b, boxes_b = [], [], [], []  # for generator
                # obj_masks = []
                # change here to split background stuff and foreground objects, category less than 91 (1-90) is foreground object, 92-183 is backgroud stuff
                for object_data in self.image_id_to_objects[image_id]:
                    objs.append(object_data['category_id'])
                    x, y, w, h = object_data['bbox']
                    if flip:
                        x0 = 1 - (x0 + x1)
                    boxes.append(np.array([x, y, w, h]))

                    if object_data['category_id'] < 91:

                        objs_f.append(object_data['category_id'])
                        x, y, w, h = object_data['bbox']
                        if flip:
                            x0 = 1 - (x0 + x1)
                        boxes_f.append(np.array([x, y, w, h]))
                    else:
                        objs_b.append(object_data['category_id'] - 91)
                        x, y, w, h = object_data['bbox']
                        if flip:
                            x0 = 1 - (x0 + x1)
                        boxes_b.append(np.array([x, y, w, h]))

                objs = torch.LongTensor(objs)
                boxes = np.vstack(boxes)

                bboxes = []
                for obj, (x,y,w,h) in zip(objs, boxes):
                    class_label = self.vocab['object_idx_to_name'][obj]
                    bboxes.append([x, y, w, h, class_label])

                transformed = self.transform(image=np.array(image.convert('RGB')), bboxes=bboxes)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']

                bboxes, labels = [], []
                for x1, y1, w, h, class_label in transformed_bboxes:
                    x2 = min(x1 + w, self.image_size)
                    y2 = min(y1 + h, self.image_size)
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(class_label)

                masks = []
                for bbox in bboxes:
                    mask = bounding_box_to_mask(self.image_size, self.image_size, bbox)
                    masks.append(mask)

        return {
            'image_path': image_path,
            'image': transformed_image / 255., # normalize to [0, 1]
            'bboxes': np.array(bboxes).astype(np.int16),
            'labels': labels,
            'masks': np.array(masks).astype(np.float32),
        }

    def __getitem__(self, idx):
        try:
            sample = self.get_data_of_index(idx)
            if len(sample['labels']) == 0:
                return self.__getitem__((idx + 1) % len(self))
            return sample
        except Exception as e:
            print(e)
            return self.__getitem__((idx + 1) % len(self))

class CocoStuffBboxCaptionDataset(CocoStuffBboxDataset):
    def __init__(self, *args, empty_string=0, use_label_as_caption=False, **kwargs):
        super().__init__(*args, **kwargs)

        captions_json = os.path.join(self.root, 'annotations', 'captions_train2017.json') if not self.validation else os.path.join(self.root, 'annotations', 'captions_val2017.json')
        with open(captions_json, 'r') as IN:
            captions_data = json.load(IN)

        self.image_id_to_captions = defaultdict(list)
        for caption_data in captions_data['annotations']:
            image_id = caption_data['image_id']
            self.image_id_to_captions[image_id].append(caption_data['caption'])

        self.empty_string = empty_string # the ration to use empty string instead of actual caption
        self.use_label_as_caption = use_label_as_caption

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        image_id = self.image_ids[index]
        if np.random.rand() > self.empty_string:
            if self.use_label_as_caption:
                caption = '; '.join(res_dict['labels'])
            else:
                captions = self.image_id_to_captions[image_id]
                caption = np.random.choice(captions)
        else:
            caption = ''
        res_dict['caption'] = caption
        return res_dict


class CocoStuffBboxCaptionDatasetPaddingVersion(CocoStuffBboxCaptionDataset):
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

    def set_image_size(self, image_size):
        height = width = image_size
        self.transform_albumatation = A.Compose([
            A.Resize(height=height, width=width, p=1),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='coco'))
        self.image_size = image_size

    def __getitem__(self, index):
        res = super().__getitem__(index)
        res['pad_size'] = self.pad_size
        res['pad_side'] = self.pad_side
        return res