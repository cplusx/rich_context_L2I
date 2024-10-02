# Code release for [Rethinking The Training And Evaluation of Rich-Context Layout-to-Image Generation](https://www.arxiv.org/pdf/2409.04847)

Working in progress... More contents coming soon...
See following instructions for preview experience

## Download Pretrained Checkpoint from HuggingFace
```
wget https://huggingface.co/cplusx/rich_context_sdxl/resolve/main/rich_context_sdxl_e580.pt
```

## Initialize the Foundational Model Weights for the UNet
```
python misc_utils/init_regional_unet.py --model_path hotshotco/SDXL-512 --save_path regional_attn_sdxl_512_init_weights --model_type ca
```

# Inference
Please see the `inference.ipynb`


This working is done during author's internship at Amazon. This codebase is a reimplementation of the work on paper. 