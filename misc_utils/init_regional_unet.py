'''
Example usage
python misc_utils/init_regional_unet.py --model_path hotshotco/SDXL-512 --save_path regional_attn_sdxl_512_init_weights --model_type ca
python misc_utils/init_regional_unet.py --model_path runwayml/stable-diffusion-v1-5 --save_path regional_attn_sd15_init_weights --model_type ca
'''
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from regional_attention.diffusers_unet import UNet2DConditionModel
import argparse

parser = argparse.ArgumentParser(description='Script to initialize weights for regional attention model')
parser.add_argument('--model_path', type=str, help='Path to the pretrained model')
parser.add_argument('--save_path', type=str, help='Path to save the initialized weights')
parser.add_argument('--model_type', type=str, help='Type of the model')

args = parser.parse_args()

unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder='unet')
unet_with_regional_attn_config = dict(unet.config)
unet_with_regional_attn_config['attention_type'] = 'regional'
unet_with_regional_attn_config['_use_default_values'] = []

if args.model_type == 'ca':
    fuser_cls = "regional_attention.regional_attention.RegionalCrossAttention"
elif args.model_type == 'casa':
    fuser_cls = "regional_attention.regional_attention.RegionalCrossAndSelfAttention"
unet_with_regional_attn_config['position_net_cls'] = "regional_attention.regional_attention.TextEmbeddingNetV2"
unet_with_regional_attn_config['fuser_cls'] = fuser_cls
unet_with_regional_attn = UNet2DConditionModel(**unet_with_regional_attn_config)
unet_sd = unet.state_dict()
unet_with_regional_attn_config = unet_with_regional_attn.state_dict()
for k in unet_with_regional_attn_config.keys():
    if k in unet_sd:
        unet_with_regional_attn_config[k] = unet_sd[k]
unet_with_regional_attn.load_state_dict(unet_with_regional_attn_config)
unet_with_regional_attn.save_pretrained(args.save_path)
