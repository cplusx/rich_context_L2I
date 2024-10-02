import torch
from torch.utils.data import Dataset
from misc_utils.model_utils import instantiate_from_config
import warnings
import numpy as np
from regional_attention.region_reorganization import region_reorganization
import albumentations as A

# Suppress specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

stacked_tensor_batching = lambda batch, key: torch.stack([torch.tensor(b[key]) for b in batch])
tensor_batching = lambda batch, key: [torch.tensor(b[key]) for b in batch]
list_batching = lambda batch, key : [b[key] for b in batch]

def custom_collate_fn(batch):
    res = {
        'image_path': list_batching(batch, 'image_path'),
        'image': stacked_tensor_batching(batch, 'image'),
        'bboxes': tensor_batching(batch, 'bboxes'),
        'labels': list_batching(batch, 'labels'),
        'masks': tensor_batching(batch, 'masks'),
        'caption': list_batching(batch, 'caption'),
        'reorg_masks': tensor_batching(batch, 'reorg_masks') if 'reorg_masks' in batch[0] else None,
        'reorg_union_masks': tensor_batching(batch, 'reorg_union_masks') if 'reorg_union_masks' in batch[0] else None,
        'reorg_labels': list_batching(batch, 'reorg_labels') if 'reorg_labels' in batch[0] else None,
        'reorg_bboxes': [[torch.tensor(b_) for b_ in b['reorg_bboxes']] for b in batch] if 'reorg_bboxes' in batch[0] else None,
        'sep_token': batch[0]['sep_token'] if 'sep_token' in batch[0] else None,
        'reorg_ious': list_batching(batch, 'reorg_ious') if 'reorg_ious' in batch[0] else None,
    }
    return res


def bounding_box_to_mask(height, width, bbox):
    mask = np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    mask[y1:y2, x1:x2] = 1
    return mask

class BoundingBoxAndMaskAugmentation:
    def __init__(self, image_size):
        self.image_size = image_size
        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=image_size),
            A.RandomCrop(width=image_size, height=image_size),
            A.HorizontalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['indices', 'labels'], min_visibility=0.3))

    def __call__(self, image, bboxes, labels, masks, indices=None):
        h, w = image.shape[:2]
        bboxes_fix_out_of_border = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            if x1 >= x2: 
                x2 = x1 + 1
                x1 = x1 - 1
            if y1 >= y2:
                y2 = y1 + 1
                y1 = y1 - 1
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            bboxes_fix_out_of_border.append([x1, y1, x2, y2])
        bboxes = bboxes_fix_out_of_border
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels, masks=masks, indices=indices)
        return transformed['image'], transformed['bboxes'], transformed['labels'], transformed['masks'], transformed['indices']


'''
Difference between JointDataset and ConcatDataset
JointDataset: samples from multiple datasets with different sampling rates, but it cannot ensure the same index will always return the same sample
ConcatDataset: simple concatenation of datasets without sampling rates, it ensures the same index will always return the same sample
'''

class JointDataset(Dataset):
    def __init__(self, subdatasets, sampling_rates=None, mode='train'):
        self.datasets = []
        for dataset_config in subdatasets:
            dataset = instantiate_from_config(dataset_config)
            self.datasets.append(dataset)
        self.sampling_rates = sampling_rates if sampling_rates is not None else [1.] * len(self.datasets)

        # Calculate total number of samples and sampling weights
        # self.total_samples = int(sum([len(d) * r for d, r in zip(self.datasets, self.sampling_rates)]))
        # self.weights = [len(d) * r / self.total_samples for d, r in zip(self.datasets, self.sampling_rates)]
        self.total_samples = int(sum([len(d) for d in self.datasets]))
        self.weights = [r for r in self.sampling_rates]

        self.mode = mode
        print('INFO: total number of samples in JointDataset is {}'.format(self.total_samples))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which dataset to sample from
        dataset_idx = torch.multinomial(torch.tensor(self.weights), 1).item()
        dataset = self.datasets[dataset_idx]

        # Sample from the chosen dataset
        sample_idx = torch.randint(len(dataset), (1,)).item() # if self.mode == 'train' else idx % len(dataset)
        return dataset[sample_idx]

class ConcatDataset(Dataset):
    # simple concatenation of datasets without sampling rates
    def __init__(self, subdatasets, mode='train'):
        self.datasets = []
        for dataset_config in subdatasets:
            dataset = instantiate_from_config(dataset_config)
            self.datasets.append(dataset)

        # Calculate total number of samples and sampling weights
        self.total_samples = int(sum([len(d) for d in self.datasets]))
        self.sample_cumsum = np.cumsum([0] + [len(d) for d in self.datasets])

        self.mode = mode
        print('INFO: total number of samples in ConcatDataset is {}'.format(self.total_samples))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which dataset to sample from
        dataset_idx = np.argmax(idx < self.sample_cumsum) - 1
        dataset = self.datasets[dataset_idx]

        # Sample from the chosen dataset
        sample_idx = idx - self.sample_cumsum[dataset_idx]
        return dataset[sample_idx]

class JointDatasetWithRegionReorg(JointDataset):
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        height, width = res['image'].shape[-2:]
        if len(res['bboxes'].shape) <= 1:
            # some samples have no objects, skip them
            return self.__getitem__(idx + 1)
        reorg_dict = region_reorganization(height, width, res['bboxes'], res['labels'])
        res.update(reorg_dict)
        res['sep_token'] = ';'

        return res

class ConcatDatasetWithRegionReorg(ConcatDataset):
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        height, width = res['image'].shape[-2:]
        if len(res['bboxes'].shape) <= 1:
            return self.__getitem__(idx + 1)
        reorg_dict = region_reorganization(height, width, res['bboxes'], res['labels'])
        res.update(reorg_dict)
        res['sep_token'] = ';'

        return res
