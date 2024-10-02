import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('-i', '--input_file', help='Input file path') 
parser.add_argument('-o', '--output_file', help='Output file path') 
args = parser.parse_args()

input_file = args.input_file 
output_file = args.output_file

import torch 
ckpt = torch.load(input_file, map_location='cpu') 
ckpt['module'] 
unet_sd = {k.replace('unet.', ''): v for k, v in ckpt['module'].items() if 'unet' in k} 
pn_sd = {k: v for k, v in unet_sd.items() if 'position' in k} 
fuser_sd = {k: v for k, v in unet_sd.items() if 'fuser' in k} 
pn_and_fuser_sd = {**pn_sd, **fuser_sd} 
pn_and_fuser_sd = {k.replace('null_positive_feature', 'null_text_feature'): v for k, v in pn_and_fuser_sd.items()}
torch.save(pn_and_fuser_sd, output_file)
