import os
from misc_utils.image_annotator import ImageAnnotator
import numpy as np
import torch
from diffusion_dataset.LAION_synthetic import LAIONSyntheticEvalDataset
from diffusion_dataset.RCCOCO import RichContextCOCOEvalDataset
from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
from regional_attention.diffusers_unet import UNet2DConditionModel
from pipelines.sdxl_pipeline import StableDiffusionXLRegionalAttnPipeline
from pipelines.sd_pipeline import StableDiffusionRegionalAttnPipeline
from omegaconf import OmegaConf
import argparse
import cv2
from PIL import Image
from tqdm import tqdm

NEGATIVE_PROMPTS = 'out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'


def plot_and_save_figure(ref_image, generated, generated_with_bboxes, caption=None, save_path=None, non_refined=None):
    import os
    import textwrap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(ref_image); ax[0].axis('off'); ax[0].set_title('Reference Image')
    ax[1].imshow(generated); ax[1].axis('off'); ax[1].set_title('Generated Image')
    ax[2].imshow(generated_with_bboxes); ax[2].axis('off'); ax[2].set_title('With Bounding Boxes')
    plt.tight_layout()
    # Save or display the figure
    if caption:
        wrapped_caption = textwrap.fill(caption, width=100)  # Adjust width as needed
        plt.suptitle(wrapped_caption, fontsize=12)  # Adjust fontsize as needed

    if save_path:
        if os.path.dirname(save_path) != '':
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(
            save_path.replace('.jpg', '.png'), 
            cv2.cvtColor((generated * 255.).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        if non_refined is not None:
            cv2.imwrite(
                save_path.replace('.jpg', '_non_refined.png'), 
                cv2.cvtColor((non_refined * 255.).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
        try:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  # Close the figure after saving to free up memory
        except Exception as e:
            print(f"Error saving figure: {e}")
    else:
        plt.show()

def get_dataset(dataset_name, image_size):
    if dataset_name == 'laion':
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
            image_size=image_size
        )
    elif dataset_name == 'rccoco':
        dataset_dir = '../DetailedSD/data/coco_2017'
        label_dir='/home/ubuntu/cjiaxin_16T/dataset-generation/rccoco_val_768'
        if not os.path.exists(label_dir):
            label_dir='/home/ubuntu/instdiff_data_generation/rccoco_val_768'
        dataset = RichContextCOCOEvalDataset(
            dataset_dir, 
            label_dir=label_dir,
            image_size=image_size,
            image_size_when_labeling=768,
            split='val',
        )
    return dataset


def get_pipe(version, config):
    unet_config = config.unet.params
    unet = UNet2DConditionModel.from_pretrained(
        **unet_config, torch_dtype=torch.float16,
    )
    if 'sd15' in version:
        pipe = StableDiffusionRegionalAttnPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16,
            unet = unet,
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            'madebyollin/sdxl-vae-fp16-fix'
        )

        pipe = StableDiffusionXLRegionalAttnPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16,
            unet = unet, vae=vae
        )
    pipe = pipe.to("cuda")

    refine_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    refine_pipe = refine_pipe.to('cuda')
    return pipe, refine_pipe, unet


def load_ckpt(ckpt_path, pipe):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'module' in ckpt:
        ckpt = ckpt['module']
    if list(ckpt.keys())[0].startswith('_forward_module'):
        ckpt = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
    # make old ckpt name to be compatible to new names in model
    if 'gligen' in ckpt_path:
        pass
    else:
        ckpt = {k.replace('null_positive_feature', 'null_text_feature'): v for k, v in ckpt.items()}
    unet_ckpt = {k.replace('unet.', ''): v for k, v in ckpt.items() if k.startswith('unet')}
    pipe.unet.load_state_dict(unet_ckpt)
    pipe.unet.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--num_images', type=int, default=10, help='Number of test samples')
    parser.add_argument('--start_idx', type=int, default=0, help='The starting index of the test samples')
    parser.add_argument('--guidance_scale', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--version', type=str, default='v4', help='Version')
    parser.add_argument('--save_dir', type=str, default='./eval', help='Directory to save the generated images')
    parser.add_argument('--dataset', type=str, default='laion', help='Dataset name')

    args = parser.parse_args()

    NUM_IMAGES = args.num_images
    GUIDANCE_SCALE = args.guidance_scale
    IMAGE_SIZE = args.image_size
    version = args.version
    DATASET = args.dataset

    annotator = ImageAnnotator()

    dataset = get_dataset(DATASET, IMAGE_SIZE)


    if version == 'sdxl':
        config_file = 'configs/layout_sdxl.yaml'
        ckpt_path = '../DetailedSD/experiments/bbox_emb_cc3m_v4_better_init/epoch_313.pth'
    elif version == 'sd15':
        config_file = 'configs/layout_sd15.yaml'
        ckpt_path = '../DetailedSD/experiments/layout_v4_sd14/epoch_315.pth'

    config = OmegaConf.load(config_file)
    pipe, refine_pipe, unet = get_pipe(version, config)

    epoch = os.path.basename(ckpt_path).split('.')[0].split('_')[-1]
    save_dir = os.path.join(args.save_dir, f'{DATASET}', f'{version}_{GUIDANCE_SCALE}_{epoch}/{IMAGE_SIZE}')

    load_ckpt(ckpt_path, pipe)


    num_test = min(NUM_IMAGES, len(dataset))
    for i in tqdm(range(args.start_idx, args.start_idx + num_test)):
        batch = dataset[i]
        image_path = batch['image_path']
        image_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, image_name)
        if os.path.exists(save_path.replace('.jpg', '.png')):
            continue
        ref_image = batch['image']
        if isinstance(ref_image, np.ndarray):
            ref_image = ref_image.transpose(1, 2, 0)
        elif isinstance(ref_image, torch.Tensor):
            ref_image = ref_image.cpu().numpy().transpose(1, 2, 0)
        else:
            raise ValueError(f"Unknown image type: {type(ref_image)}")
        bboxes = batch['bboxes']
        if len(bboxes) == 0:
            continue
        caption = batch['caption']
        labels = batch['labels']
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            height, width = ref_image.shape[:2]
            if 'gligen' in version:
                try:
                    generated = pipe(
                        prompt=caption,
                        negative_prompt=NEGATIVE_PROMPTS,
                        height=height, width=width,
                        gligen_boxes=bboxes / IMAGE_SIZE,
                        gligen_phrases=labels,
                        gligen_scheduled_sampling_beta=1,
                        num_inference_steps=25,
                        guidance_scale=GUIDANCE_SCALE
                    )
                except Exception as e:
                    print(f"Error generating image: {e}")
                    continue
            else:
                try:
                    generated = pipe(
                        prompt=caption,
                        negative_prompt=NEGATIVE_PROMPTS,
                        height=height, width=width,
                        boxes=bboxes,
                        labels=labels,
                        scheduled_sampling_beta=1,
                        num_inference_steps=25,
                        guidance_scale=GUIDANCE_SCALE
                    )
                except Exception as e:
                    print(f"Error generating image: {e}")
                    continue
            generated = np.array(generated[0][0]) / 255.

        use_refine = True
        if use_refine:
            non_refined = generated.copy()
            sample = np.array(generated * 255, dtype=np.uint8)
            sample = Image.fromarray(sample)
            refined_image = refine_pipe(caption, image=sample, strength=0.3, num_inference_steps=20).images[0]
            generated = np.array(refined_image) / 255.
        else:
            non_refined = None

        if 'pad_side' in batch:
            pad_side = batch['pad_side']
            pad_size = batch['pad_size']
            if pad_size > 0:
                if pad_side == 'right':
                    generated = generated[:, :-pad_size]
                    non_refined = None if non_refined is None else non_refined[:, :-pad_size]
                    ref_image = ref_image[:, :-pad_size]
                elif pad_side == 'bottom':
                    generated = generated[:-pad_size]
                    non_refined = None if non_refined is None else non_refined[:-pad_size]
                    ref_image = ref_image[:-pad_size]
        
        generated_with_bboxes = annotator.add_bboxes(generated, bboxes, labels)
        plot_and_save_figure(ref_image, generated, generated_with_bboxes, caption=caption, save_path=save_path, non_refined=non_refined)
