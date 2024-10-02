import torch
import numpy as np
from einops import rearrange

def compute_iou_between_masks(mask1, mask2):
    assert mask1.shape == mask2.shape, f'mask1.shape={mask1.shape} != mask2.shape={mask2.shape}'
    intersection = (mask1 * mask2).sum()
    union = ((mask1 + mask2) > 0).sum()
    assert union > 0, f'union={union} <= 0'
    return intersection / union

def get_union_mask(intersection_masks, region_masks):
    intersection_masks = rearrange(intersection_masks, 'n1 h w -> n1 1 h w')
    raw_mask = rearrange(region_masks, 'n2 h w -> 1 n2 h w')

    union_masks = (intersection_masks + raw_mask) > 0 # n1, n2, h, w (logical or)
    union_masks_have_overlap = (intersection_masks * raw_mask).sum(axis=(2, 3), keepdims=True) > 0 # (logical and)
    mask_is_background = union_masks_have_overlap.sum(axis=1) < 0.5 # if there is no overlap, the mask is background
    background_idx = np.where(mask_is_background)[0]
    union_masks = (union_masks * union_masks_have_overlap).sum(axis=1) > 0
    union_masks[background_idx] = True
    return union_masks.astype(np.float32)

def get_reorganized_mask(height, width, bboxes, mask_reduce_factor=8):
    N, _ = bboxes.shape
    height, width = int(height/mask_reduce_factor), int(width/mask_reduce_factor)
    region_masks = np.zeros((N, height, width), dtype=bool)

    H, W = height, width
    for j in range(N):
        x1, y1, x2, y2 = bboxes[j]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, min(W-1, x1))
        y1 = max(0, min(H-1, y1))
        if x1 >= x2:
            x2 = x1
        if y1 >= y2:
            y2 = y1
        y2 = max(0, min(H-1, y2))
        x2 = max(0, min(W-1, x2))
        region_masks[j, y1:y2, x1:x2] = True

    region_mask_transpose = rearrange(region_masks, 'n h w -> (h w) n') # l, n
    unique_regions, region_indices = np.unique(region_mask_transpose, axis=0, return_inverse=True)
    num_unique_regions = unique_regions.shape[0]

    region_indices_expanded = np.expand_dims(region_indices, axis=0).repeat(num_unique_regions, axis=0)
    j_indices = np.arange(num_unique_regions).reshape(-1, 1).repeat(region_indices_expanded.shape[1], axis=1)

    # Creating a boolean mask where each row corresponds to a unique region
    intersection_masks = (j_indices == region_indices_expanded)
    intersection_masks = rearrange(intersection_masks, 'n (h w) -> n h w', h=height, w=width)

    union_masks = get_union_mask(intersection_masks, region_masks)

    return intersection_masks, num_unique_regions, unique_regions, union_masks, region_masks

def region_reorganization(height, width, bboxes, labels, max_objs=30, mask_reduce_factor=8, max_regions=120):
    H, W = int(height/mask_reduce_factor), int(width/mask_reduce_factor)

    bboxes = bboxes / mask_reduce_factor
    N, _ = bboxes.shape
    if N > max_objs:
        print(f'WARNING: number of objects {N} > max_objs {max_objs}, truncating')
        N = max_objs
        bboxes = bboxes[:N]
        labels = labels[:N]
    labels_and_bboxes = [(labels[i], bboxes[i]) for i in range(N)]

    intersection_masks, num_unique_regions, unique_regions, union_masks, region_masks = get_reorganized_mask(height, width, bboxes, mask_reduce_factor)
    
    reorganized_labels = []
    reorganized_bboxes = []
    reorganized_ious = []
    for j in range(num_unique_regions):
        this_region_labels_and_bboxes = [labels_and_bboxes[i] for i in range(unique_regions.shape[1]) if unique_regions[j, i]] # e.g., unique_region[j, i]=[0, 1, 1], labels=[a, b, c], this_region_labels=[b, c]
        if len(this_region_labels_and_bboxes) == 0:
            this_region_labels = ''
            this_region_bboxes = torch.tensor([-W, -H, -W, -H], dtype=torch.float64).view(1, 4)
            normed_ious = [1.] # assign 1.0 to mean do not change the behavior of the xattn
        else:
            this_region_labels = '; '.join([l.replace(';', '') for l, b in this_region_labels_and_bboxes]) # ensure the labels does not contain the separator
            this_region_bboxes = torch.stack([
                torch.tensor(b) for l, b in this_region_labels_and_bboxes
            ]) # [num_objs, 4]
            this_region_intersect_masks = [region_masks[i] for i in range(unique_regions.shape[1]) if unique_regions[j, i]] # find all masks that intersect with this region
            ious = [compute_iou_between_masks(intersection_masks[j], mask) for mask in this_region_intersect_masks]
            normed_ious = [i / max(ious) for i in ious]

        # reorganized bboxes are in normalized coordinates
        this_region_bboxes[:, [0, 2]] /= W
        this_region_bboxes[:, [1, 3]] /= H
        reorganized_labels.append(this_region_labels)
        reorganized_bboxes.append(this_region_bboxes)
        reorganized_ious.append(normed_ious)

    assert len(reorganized_labels) == len(intersection_masks), f'len(reorganized_labels)={len(reorganized_labels)} != len(reorganized_mask)={len(intersection_masks)}'

    intersection_masks = intersection_masks[:max_regions]
    union_masks = union_masks[:max_regions]
    reorganized_labels = reorganized_labels[:max_regions]
    reorganized_bboxes = reorganized_bboxes[:max_regions]
    reorganized_ious = reorganized_ious[:max_regions]

    return {
        'reorg_masks': intersection_masks,
        'reorg_union_masks': union_masks,
        'reorg_labels': reorganized_labels,
        'reorg_bboxes': reorganized_bboxes,
        'reorg_ious': reorganized_ious
    }


def find_intervals(token_ids, tokenizer, sep_token_id):
    first_idx = [index for index, value in enumerate(token_ids) if value == tokenizer.bos_token_id][0]
    if tokenizer.eos_token_id in token_ids:
        last_idx = [index for index, value in enumerate(token_ids) if value == tokenizer.eos_token_id][0]
    else:
        last_idx = len(token_ids) - 1
    sep_ids_indices = [index for index, value in enumerate(token_ids) if value == sep_token_id]

    if len(sep_ids_indices) == 0:
        return [(first_idx + 1, last_idx)]

    intervals = [(first_idx+1, sep_ids_indices[0])]
    for i in range(len(sep_ids_indices)-1):
        intervals.append((sep_ids_indices[i]+1, sep_ids_indices[i+1]))
    intervals.append((sep_ids_indices[-1]+1, last_idx))
    return intervals

def prepare_train_attention_kwargs(
    reorg_bboxes, 
    reorg_labels, 
    reorg_masks, 
    reorg_union_masks, 
    reorg_ious, 
    sep_token, 
    cross_attention_dim, # should be unet.config.cross_attention_dim
    tokenizer, tokenizer_2,
    text_encoder, text_encoder_2,
    device, dtype,
    max_objs=30,
    attention_weight_scale=1.0
):
    assert len(reorg_labels) == len(reorg_masks)
    n_objs = len(reorg_labels)
    if n_objs > max_objs:
        picked_idx = np.random.choice(n_objs, max_objs, replace=False)
        reorg_labels = [reorg_labels[i] for i in picked_idx]
        reorg_masks = reorg_masks[picked_idx]
        reorg_union_masks = reorg_union_masks[picked_idx]
        reorg_ious = [reorg_ious[i] for i in picked_idx]
        n_objs = max_objs

    label_text_embeddings = torch.zeros(
        max_objs, 
        tokenizer.model_max_length,
        cross_attention_dim, 
        device=device,
        dtype=dtype
    )

    bbox_embeddings = -1 * torch.ones(max_objs, tokenizer.model_max_length, 4, device=device, dtype=dtype) # N, 77, 4
    cross_attention_mask = torch.ones(
        max_objs, 
        tokenizer.model_max_length, 
        device=device, dtype=text_encoder.dtype
    ) * -1e3
    sep_token_id = tokenizer(sep_token, add_special_tokens=False)['input_ids'][0]

    if len(reorg_labels) > 0:
        tokenizer_inputs = tokenizer(
            reorg_labels, padding='max_length', return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)
        text_embeddings = text_encoder(**tokenizer_inputs).last_hidden_state # N, 77, 768

        if (tokenizer_2 is not None) and (text_encoder_2 is not None):
            tokenizer_inputs_2 = tokenizer_2(
                reorg_labels, padding='max_length', return_tensors="pt",
                truncation=True,
                max_length=tokenizer_2.model_max_length
            ).to(device)
            text_embeddings_2 = text_encoder_2(**tokenizer_inputs_2).last_hidden_state # N, 77, 1280
            text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1) # N, 77, 2048
        label_text_embeddings[:n_objs] = text_embeddings.to(dtype)

        # prepare bbox embeddings
        for label_idx in range(n_objs):
            input_ids = tokenizer_inputs['input_ids'][label_idx]
            bbox_intervals = find_intervals(input_ids, tokenizer, sep_token_id)
            for box, (start_idx, end_idx) in zip(reorg_bboxes[label_idx], bbox_intervals):
                # print(box, start_idx, end_idx) # box here should be normalized to 0-1
                bbox_embeddings[label_idx, start_idx:end_idx] = torch.tensor(box, device=device, dtype=dtype)

        # it can be merged with bbox embeddings, but for now, we keep it separate for clear understanding
        # e.g., [-1e3, 0, -0.5, -1e3], the pre-softmax value will add this value, so we use iou-1 to ensure the maximum value is 0
        for i in range(n_objs):
            inputs_ids = tokenizer_inputs['input_ids'][i]
            ious = reorg_ious[i]
            intervals = find_intervals(inputs_ids, tokenizer, sep_token_id)
            for iou, (start_idx, end_idx) in zip(ious, intervals):
                cross_attention_mask[i, start_idx:end_idx] = (iou - 1) * attention_weight_scale
    else:
        pass
        # empty label_text_embeddings

    object_masks = torch.zeros(max_objs, device=device, dtype=dtype)
    object_masks[:n_objs] = 1

    reorg_masks_ = torch.zeros(max_objs, *reorg_masks.shape[1:], device=device, dtype=dtype)
    reorg_masks_[:n_objs] = torch.tensor(reorg_masks, device=device, dtype=dtype)

    reorg_union_masks_ = torch.zeros(max_objs, *reorg_union_masks.shape[1:], device=device, dtype=dtype)
    reorg_union_masks_[:n_objs] = torch.tensor(reorg_union_masks, device=device, dtype=dtype)

    cross_attention_mask[n_objs:] = 0.

    return {
        'reorg_masks': reorg_masks_.unsqueeze(0),
        'reorg_union_masks': reorg_union_masks_.unsqueeze(0),
        'bbox_embeddings': bbox_embeddings.unsqueeze(0),
        'object_masks': object_masks.unsqueeze(0), # indicate whether it is an object (1) or not (0, for padding to the same length)
        'text_embeddings': label_text_embeddings.unsqueeze(0),
        'cross_attention_mask': cross_attention_mask.unsqueeze(0)
    }

def prepare_empty_attention_kwargs(
    mask, 
    cross_attention_dim,
    tokenizer,
    text_encoder,
    device, dtype,
    max_objs=30
):
    reorg_masks = torch.zeros(max_objs, *mask.shape[1:], device=device, dtype=dtype)
    reorg_union_masks = torch.zeros(max_objs, *mask.shape[1:], device=device, dtype=dtype)
    reorg_union_masks[0] = torch.ones(mask[0].shape)
    bbox_embeddings = -1 * torch.ones(max_objs, tokenizer.model_max_length, 4, device=device, dtype=dtype) # N, 77, 4
    cross_attention_mask = torch.zeros(
        max_objs, 
        tokenizer.model_max_length, 
        device=device, dtype=text_encoder.dtype
    )
    object_masks = torch.zeros(max_objs, device=device, dtype=dtype)
    label_text_embeddings = torch.zeros(
        max_objs, 
        tokenizer.model_max_length,
        cross_attention_dim, 
        device=device,
        dtype=dtype
    )
    return {
        'reorg_masks': reorg_masks.unsqueeze(0),
        'reorg_union_masks': reorg_union_masks.unsqueeze(0),
        'bbox_embeddings': bbox_embeddings.unsqueeze(0),
        'object_masks': object_masks.unsqueeze(0),
        'text_embeddings': label_text_embeddings.unsqueeze(0),
        'cross_attention_mask': cross_attention_mask.unsqueeze(0)
    }

@torch.no_grad()
def prepare_pipeline_attention_kwargs(
    height, width,
    boxes, labels,
    device, dtype,
    batch_size, num_images_per_prompt,
    cross_attention_dim,
    tokenizer, tokenizer_2,
    text_encoder, text_encoder_2,
    sep_token=';',
    max_objs=30,
    attention_weight_scale=1.0,
    do_classifier_free_guidance=True,
    **region_reorganization_kwargs,
):
    reorg_dict = region_reorganization(
        height, width, boxes, labels, max_objs=max_objs, **region_reorganization_kwargs
    )
    reorg_bboxes = reorg_dict['reorg_bboxes']
    reorg_labels = reorg_dict['reorg_labels']
    reorg_masks = reorg_dict['reorg_masks']
    reorg_union_masks = reorg_dict['reorg_union_masks']
    reorg_ious = reorg_dict['reorg_ious']

    cond_attn_kwargs = prepare_train_attention_kwargs(
        reorg_bboxes, 
        reorg_labels, 
        reorg_masks, 
        reorg_union_masks, 
        reorg_ious, 
        sep_token, 
        cross_attention_dim, # should be unet.config.cross_attention_dim
        tokenizer, tokenizer_2,
        text_encoder, text_encoder_2,
        device, dtype,
        max_objs=max_objs,
        attention_weight_scale=attention_weight_scale
    )

    repeat_batch = batch_size * num_images_per_prompt

    cond_attn_kwargs = {
        k: v.repeat(
            repeat_batch, *list([1]*len(v.shape[1:]))
        ) for k, v in cond_attn_kwargs.items()
    }

    if do_classifier_free_guidance:
        negative_attn_kwargs = prepare_empty_attention_kwargs(
            reorg_masks, 
            cross_attention_dim,
            tokenizer, text_encoder,
            device, dtype,
            max_objs=max_objs
        )
        negative_attn_kwargs = {
            k: v.repeat(
                repeat_batch, *list([1]*len(v.shape[1:]))
            ) for k, v in negative_attn_kwargs.items()
        }

        cond_attn_kwargs = {
            k: torch.cat([negative_attn_kwargs[k], cond_attn_kwargs[k]], dim=0) for k in cond_attn_kwargs
        }

    return cond_attn_kwargs

# below is for ablation with instdiff, can be removed
def prepare_train_attention_kwargs_instdiff(
    reorg_bboxes, 
    reorg_labels, 
    reorg_masks, 
    reorg_union_masks, 
    reorg_ious, 
    sep_token, 
    cross_attention_dim, # should be unet.config.cross_attention_dim
    tokenizer, tokenizer_2,
    text_encoder, text_encoder_2,
    device, dtype,
    max_objs=30,
):
    assert len(reorg_labels) == len(reorg_masks)
    n_objs = len(reorg_labels)
    if n_objs > max_objs:
        picked_idx = np.random.choice(n_objs, max_objs, replace=False)
        reorg_labels = [reorg_labels[i] for i in picked_idx]
        reorg_masks = reorg_masks[picked_idx]
        reorg_union_masks = reorg_union_masks[picked_idx]
        reorg_ious = [reorg_ious[i] for i in picked_idx]
        n_objs = max_objs

    label_text_embeddings = torch.zeros(
        max_objs, 
        1, #tokenizer.model_max_length,
        cross_attention_dim, 
        device=device,
        dtype=dtype
    )

    bbox_embeddings = -1 * torch.ones(max_objs, tokenizer.model_max_length, 4, device=device, dtype=dtype) # N, 77, 4
    sep_token_id = tokenizer(sep_token, add_special_tokens=False)['input_ids'][0]

    if len(reorg_labels) > 0:
        tokenizer_inputs = tokenizer(
            reorg_labels, padding='max_length', return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)
        text_embeddings = text_encoder(**tokenizer_inputs).pooler_output # N, 768
        text_embeddings = rearrange(text_embeddings, 'n d -> n 1 d')

        label_text_embeddings[:n_objs] = text_embeddings.to(dtype)

        # prepare bbox embeddings
        for label_idx in range(n_objs):
            input_ids = tokenizer_inputs['input_ids'][label_idx]
            bbox_intervals = find_intervals(input_ids, tokenizer, sep_token_id)
            for box, (start_idx, end_idx) in zip(reorg_bboxes[label_idx], bbox_intervals):
                # print(box, start_idx, end_idx) # box here should be normalized to 0-1
                bbox_embeddings[label_idx, start_idx:end_idx] = torch.tensor(box, device=device, dtype=dtype)
    else:
        pass
        # empty label_text_embeddings

    bbox_embeddings = bbox_embeddings[:, :1] # only the first token is used for instdiff

    object_masks = torch.zeros(max_objs, device=device, dtype=dtype)
    object_masks[:n_objs] = 1

    reorg_masks_ = torch.zeros(max_objs, *reorg_masks.shape[1:], device=device, dtype=dtype)
    reorg_masks_[:n_objs] = torch.tensor(reorg_masks, device=device, dtype=dtype)

    reorg_union_masks_ = torch.zeros(max_objs, *reorg_union_masks.shape[1:], device=device, dtype=dtype)
    reorg_union_masks_[:n_objs] = torch.tensor(reorg_union_masks, device=device, dtype=dtype)


    return {
        'reorg_masks': reorg_masks_.unsqueeze(0),
        'reorg_union_masks': reorg_union_masks_.unsqueeze(0),
        'bbox_embeddings': bbox_embeddings.unsqueeze(0),
        'object_masks': object_masks.unsqueeze(0), # indicate whether it is an object (1) or not (0, for padding to the same length)
        'text_embeddings': label_text_embeddings.unsqueeze(0),
    }

def prepare_empty_attention_kwargs_instdiff(
    mask, 
    cross_attention_dim,
    tokenizer,
    text_encoder,
    device, dtype,
    max_objs=30
):
    reorg_masks = torch.zeros(max_objs, *mask.shape[1:], device=device, dtype=dtype)
    reorg_union_masks = torch.zeros(max_objs, *mask.shape[1:], device=device, dtype=dtype)
    reorg_union_masks[0] = torch.ones(mask[0].shape)
    bbox_embeddings = -1 * torch.ones(max_objs, 1, 4, device=device, dtype=dtype)
    object_masks = torch.zeros(max_objs, device=device, dtype=dtype)
    label_text_embeddings = torch.zeros(
        max_objs, 
        1,
        cross_attention_dim, 
        device=device,
        dtype=dtype
    )
    return {
        'reorg_masks': reorg_masks.unsqueeze(0),
        'reorg_union_masks': reorg_union_masks.unsqueeze(0),
        'bbox_embeddings': bbox_embeddings.unsqueeze(0),
        'object_masks': object_masks.unsqueeze(0),
        'text_embeddings': label_text_embeddings.unsqueeze(0),
    }

@torch.no_grad()
def prepare_instdiff_pipeline_attention_kwargs(
    height, width,
    boxes, labels,
    device, dtype,
    batch_size, num_images_per_prompt,
    cross_attention_dim,
    tokenizer, tokenizer_2,
    text_encoder, text_encoder_2,
    sep_token=';',
    max_objs=30,
    attention_weight_scale=1.0,
    do_classifier_free_guidance=True,
    **region_reorganization_kwargs,
):
    reorg_dict = region_reorganization(
        height, width, boxes, labels, max_objs=max_objs, **region_reorganization_kwargs
    )
    reorg_bboxes = reorg_dict['reorg_bboxes']
    reorg_labels = reorg_dict['reorg_labels']
    reorg_masks = reorg_dict['reorg_masks']
    reorg_union_masks = reorg_dict['reorg_union_masks']
    reorg_ious = reorg_dict['reorg_ious']

    cond_attn_kwargs = prepare_train_attention_kwargs_instdiff(
        reorg_bboxes, 
        reorg_labels, 
        reorg_masks, 
        reorg_union_masks, 
        reorg_ious, 
        sep_token, 
        cross_attention_dim, # should be unet.config.cross_attention_dim
        tokenizer, tokenizer_2,
        text_encoder, text_encoder_2,
        device, dtype,
        max_objs=max_objs,
    )

    repeat_batch = batch_size * num_images_per_prompt

    cond_attn_kwargs = {
        k: v.repeat(
            repeat_batch, *list([1]*len(v.shape[1:]))
        ) for k, v in cond_attn_kwargs.items()
    }

    if do_classifier_free_guidance:
        negative_attn_kwargs = prepare_empty_attention_kwargs_instdiff(
            reorg_masks, 
            cross_attention_dim,
            tokenizer, text_encoder,
            device, dtype,
            max_objs=max_objs
        )
        negative_attn_kwargs = {
            k: v.repeat(
                repeat_batch, *list([1]*len(v.shape[1:]))
            ) for k, v in negative_attn_kwargs.items()
        }

        cond_attn_kwargs = {
            k: torch.cat([negative_attn_kwargs[k], cond_attn_kwargs[k]], dim=0) for k in cond_attn_kwargs
        }

    return cond_attn_kwargs
