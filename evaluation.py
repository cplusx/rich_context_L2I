import os
import numpy as np
from misc_utils.eval_utils import CropCLIPScore, SAMIoU, CropPickScore
from diffusion_dataset.RCCOCO import RichContextCOCOEvalDataset
from diffusion_dataset.LAION_synthetic import LAIONSyntheticEvalDataset
from tqdm import tqdm
import cv2
import argparse
import json

def load_gligen_image(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        return cv2.imread(image_path)[..., ::-1]
    return None

def load_boxdiff_image(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        return cv2.imread(image_path)[..., ::-1]
    return None

def load_instdiff_image(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name, 'image.png')
    if os.path.exists(image_path):
        image = cv2.imread(image_path)[..., ::-1]
        return image
    else:
        print(f"Image not found: {image_path}")
    return None

def load_ours_image(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name.replace('.jpg', '.png'))
    if os.path.exists(image_path):
        return cv2.imread(image_path)[..., ::-1]
    return None

def load_ours_image_no_refine(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name.replace('.jpg', '_non_refined.png'))
    if os.path.exists(image_path):
        return cv2.imread(image_path)[..., ::-1]
    else:
        print(f"Image not found: {image_path}")
    return None

def evaluate_one_image(image_np, bboxes, labels):
    all_simi = []
    all_pickscore = []
    ious = []
    object_size_clip_simi = []
    object_size_pick_score = []
    object_size_sam_iou = []
    for input_box, label in zip(bboxes, labels):
        input_box = input_box.astype(np.int32)
        obj_size = int((input_box[2] - input_box[0]) * (input_box[3] - input_box[1]))
        # Compute similarity score
        simi = clip_scorer.compute_score_wbbox(image_np, label, input_box)
        if simi is not None:
            all_simi.append(simi)
            object_size_clip_simi.append(obj_size)

        # compute pick score
        pickscore = pick_scorer.compute_score_wbbox(image_np, label, input_box)
        if pickscore is not None:
            all_pickscore.append(pickscore)
            object_size_pick_score.append(obj_size)

        # Compute IoU score
        sam_iou.set_image(image_np)
        sam_mask, score = sam_iou.get_mask_from_bbox(input_box)
        if score > 0.5:
            try:
                sam_box = sam_iou.get_bbox_from_mask(sam_mask)
                iou = sam_iou.compute_iou(input_box, sam_box)
                ious.append(iou)
            except:
                ious.append(0)
        else:
            ious.append(0)
        object_size_sam_iou.append(obj_size)

    mean_simi = np.mean(all_simi) if len(all_simi) > 0 else None
    mean_pick_score = np.mean(all_pickscore) if len(all_pickscore) > 0 else None
    mean_iou = np.mean(ious) if len(ious) > 0 else None

    return mean_simi, all_simi, mean_pick_score, all_pickscore, mean_iou, ious, object_size_clip_simi, object_size_pick_score, object_size_sam_iou

def save_to_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {save_path}")

image_read_funcs = {
    # sdxl
    'v4_4.0_nr_313': load_ours_image_no_refine,
    'v4_4.5_nr_313': load_ours_image_no_refine,
    'v4_6.0_nr_313': load_ours_image_no_refine,
    # sd15
    'v4_sd15_7.5_nr_315': load_ours_image_no_refine,
    # rebuttal
    'no_refine':load_ours_image_no_refine,
    'refine': load_ours_image,
}

if __name__ == '__main__':
    '''
    v4: python evaluation.py --method v4_4.0 --image_dir /home/ubuntu/cjiaxin_16T/DetailedSD/eval/v4_4.0 --image_size 512 --save_dir tmp --num_images 10000 
    sd15 tuned: CUDA_VISIBLE_DEVICES=0 python evaluation.py --method v4_sd15_7.5 --image_dir eval/v4_sd15_7.5 --image_size 512 --save_dir tmp --num_images 10000
    '''

    parser = argparse.ArgumentParser(description='Image evaluation')
    parser.add_argument('--method', type=str, default='v4', help='Method name')
    parser.add_argument('--image_dir', type=str, default='', help='Image directory')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--save_dir', type=str, default='tmp', help='Save directory')
    parser.add_argument('--num_images', type=int, default=20, help='Number of images to evaluate')
    parser.add_argument('--dataset', type=str, default='laion', help='Dataset directory')

    args = parser.parse_args()

    method_name = args.method
    image_dir = f'{args.image_dir}/{args.image_size}'
    image_size = args.image_size
    num_images = args.num_images
    dataset = args.dataset
    save_dir = f'{args.save_dir}/{dataset}/{method_name}_{image_size}'
    os.makedirs(save_dir, exist_ok=True)

    image_read_fn = image_read_funcs[method_name]

    clip_scorer = CropCLIPScore()
    pick_scorer = CropPickScore()
    sam_iou = SAMIoU()

    if dataset == 'laion':
        dataset_dir = '/home/ubuntu/cjiaxin_16T/LAION400M512'
        synthetic_data_dir='/home/ubuntu/cjiaxin_16T/dataset-generation/generated_data_512'
        if not os.path.exists(dataset_dir):
            dataset_dir = '/home/ubuntu/LAION400M512'
            synthetic_data_dir='/home/ubuntu/instdiff_data_generation/generated_data_512'
        dataset = LAIONSyntheticEvalDataset(
            dataset_dir, 
            synthetic_data_dir=synthetic_data_dir,
            force_regenerate_meta=False,
            meta_name='meta_eval.json', 
            image_size=512, # this is fixed size
        )
    elif dataset == 'rccoco':
        dataset_dir = 'data/coco_2017'
        label_dir='/home/ubuntu/cjiaxin_16T/dataset-generation/rccoco_val_768'
        if not os.path.exists(label_dir):
            label_dir='/home/ubuntu/instdiff_data_generation/rccoco_val_768'
        dataset = RichContextCOCOEvalDataset(
            dataset_dir, 
            label_dir=label_dir,
            image_size=512,
            image_size_when_labeling=768,
            split='val',
        )

    all_msimi = []
    all_simi = []
    all_object_sizes_clip_simi = []
    all_mpick = []
    all_pickscore = []
    all_object_sizes_pick_score = []
    all_miou = []
    all_iou = []
    all_object_sizes_sam_iou = []
    detailed_res = {}
    for batch_idx in tqdm(range(min(num_images, len(dataset)))):
        batch = dataset[batch_idx]
        # image = batch['image']
        # image_np = (image.transpose(1, 2, 0) * 255).astype('uint8')

        image_path = batch['image_path']
        image_name = os.path.basename(image_path)
        image_np = image_read_fn(image_dir, image_name)
        if image_np is None:
            continue

        bboxes = batch['bboxes'] * 1. / 512 * image_size # rescale bbox
        labels = batch['labels']
        msimi, simis, mpick, picks, miou, ious, object_sizes_clip_score, object_size_pick_score, object_sizes_sam_iou = evaluate_one_image(image_np, bboxes, labels)
        if msimi is not None:
            all_msimi.append(msimi)
            all_simi.extend(simis)
            all_object_sizes_clip_simi.extend(object_sizes_clip_score)
        if mpick is not None:
            all_mpick.append(mpick)
            all_pickscore.extend(picks)
            all_object_sizes_pick_score.extend(object_size_pick_score)
        if miou is not None:
            all_miou.append(miou)
            all_iou.extend(ious)
            all_object_sizes_sam_iou.extend(object_sizes_sam_iou)

        detailed_res[image_name] = {
            'msimi': msimi,
            'simi': simis,
            'object_sizes_clip_simi': object_sizes_clip_score,
            'mpick': mpick,
            'picks': picks,
            'object_sizes_pick_score': object_size_pick_score,
            'miou': miou,
            'iou': ious,
            'object_sizes_sam_iou': object_sizes_sam_iou
        }

        if batch_idx % 1000 == 0:

            res = {
                'msimi': float(np.mean(all_msimi)),
                'simi': all_simi,
                'object_sizes_clip_simi': all_object_sizes_clip_simi,
                'mpick': float(np.mean(all_mpick)),
                'picks': all_pickscore,
                'object_sizes_pick_score': all_object_sizes_pick_score,
                'miou': float(np.mean(all_miou)),
                'iou': all_iou,
                'object_sizes_sam_iou': all_object_sizes_sam_iou
            }
            save_to_json(res, os.path.join(save_dir, f'results_{batch_idx}.json'))
            save_to_json(detailed_res, os.path.join(save_dir, f'detailed_results_{batch_idx}.json'))

    res = {
        'msimi': float(np.mean(all_msimi)),
        'simi': all_simi,
        'object_sizes_clip_simi': all_object_sizes_clip_simi,
        'mpick': float(np.mean(all_mpick)),
        'picks': all_pickscore,
        'object_sizes_pick_score': all_object_sizes_pick_score,
        'miou': float(np.mean(all_miou)),
        'iou': all_iou,
        'object_sizes_sam_iou': all_object_sizes_sam_iou
    }
    save_to_json(res, os.path.join(save_dir, f'results_{num_images}.json'))
    save_to_json(detailed_res, os.path.join(save_dir, f'detailed_results_{num_images}.json'))