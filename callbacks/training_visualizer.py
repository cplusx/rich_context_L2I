import os
from typing import Any
import pytorch_lightning as pl
import torch
import torchvision
import cv2
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import wandb
from misc_utils.image_annotator import ImageAnnotator

def unnorm(x):
    '''convert from range [-1, 1] to [0, 1]'''
    return (x+1) / 2

def clip_image(x, min=0., max=1.):
    return torch.clamp(x, min=min, max=max)

def format_dtype_and_shape(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        if len(x.shape) == 4 and x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        x = x.detach().cpu().numpy()
    return x

def tensor2image(x):
    x = x.float() # handle bf16
    '''convert 4D (b, dim, h, w) pytorch tensor to wandb Image class'''
    grid_img = torchvision.utils.make_grid(
        x, nrow=4
    ).permute(1, 2, 0).detach().cpu().numpy()
    img = wandb.Image(
        grid_img
    )
    return img

class InstructedP2PTrainingLogger(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger=None,
        max_num_images: int=16,
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.max_num_images = max_num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 1000 == 0:
            input_image = tensor2image(clip_image(
                batch['image'][:self.max_num_images]
            ))
            model_output = tensor2image(clip_image(
                outputs['x_0_hat'][:self.max_num_images]
            ))
            # masks = torch.stack(batch['masks']).permute(1, 0, 2, 3)[:self.max_num_images]
            # masks_with_border = masks.clone()
            # masks_with_border[:, :, 0, :] = 1
            # masks_with_border[:, :, -1, :] = 1
            # masks_with_border[:, :, :, 0] = 1
            # masks_with_border[:, :, :, -1] = 1
            # masks = tensor2image(
            #     masks_with_border
            # )
            self.wandb_logger.experiment.log({
                'train/input_image': input_image,
                'train/model_output': model_output,
                # 'train/masks': masks,
            })

            labels = batch['labels'][0]
            columns = ['caption', 'labels']
            data = [
                [
                    batch['caption'][0], 
                    '|'.join(labels)
                ]
            ]
            self.wandb_logger.log_text(key='train/text', columns=columns, data=data)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            input_image = tensor2image(clip_image(
                batch['image'][:self.max_num_images]
            ))
            model_output = tensor2image(clip_image(
                torch.from_numpy(outputs[:self.max_num_images]).permute(0, 3, 1, 2)
            ))
            # batch['masks'] shape: [num_masks, H, W], TODO, use rearrange
            masks = torch.stack(batch['masks']).permute(1, 0, 2, 3)[:self.max_num_images]
            masks_with_border = masks.clone()
            masks_with_border[:, :, 0, :] = 1
            masks_with_border[:, :, -1, :] = 1
            masks_with_border[:, :, :, 0] = 1
            masks_with_border[:, :, :, -1] = 1
            masks = tensor2image(
                masks_with_border
            )

            labels = batch['labels'][0]

            # current only support batch size 1
            generated_np = outputs[:self.max_num_images] # (b, h, w, 3)
            # print(batch['bboxes'][0].shape)
            # print(batch['masks'][0].shape)
            # print(labels)
            annotator = ImageAnnotator()
            image_with_bboxes = annotator.add_bboxes(
                image=generated_np[0], 
                bboxes=batch['bboxes'][0].float().cpu().numpy(), 
                labels=labels,
            )
            image_with_bboxes = tensor2image(
                torch.from_numpy(image_with_bboxes).permute(2, 0, 1)[None]
            )
            image_with_masks = annotator.add_masks(
                image=generated_np[0], 
                masks=batch['masks'][0].float().cpu().numpy(), 
                labels=labels,
            )
            image_with_masks = tensor2image(
                torch.from_numpy(image_with_masks).permute(2, 0, 1)[None]
            )

            self.wandb_logger.experiment.log({
                'val/input_image': input_image,
                'val/model_output': model_output,
                'val/masks': masks,
                'val/image_with_bboxes': image_with_bboxes,
                'val/image_with_masks': image_with_masks,
            })

            columns = ['caption', 'labels']
            data = [
                [
                    batch['caption'][0], 
                    '|'.join(labels)
                ]
            ]
            self.wandb_logger.log_text(key='val/text', columns=columns, data=data)