## Code release for [Rethinking The Training And Evaluation of Rich-Context Layout-to-Image Generation (NeurIPS 2024)](https://www.arxiv.org/pdf/2409.04847)

Working in progress... More contents coming soon...
See following instructions for preview experience


## Inference
### Initialize the Foundational Model Weights for the UNet
```
python misc_utils/init_regional_unet.py --model_path hotshotco/SDXL-512 --save_path regional_attn_sdxl_512_init_weights --model_type ca
```
Please see the `inference.ipynb` for using the Rich-context L2I with `diffusers`.

## Generate Synthetic Data
Coming soon

## Training Your Own Model
Coming soon

## Citation
If you find this work useful, please cite this work
```
@inproceedings{rich_context_l2i,
  title = {Rethinking The Training And Evaluation of Rich-Context Layout-to-Image Generation},
  author = {Cheng, Jiaxin and Zhao, Zixu and He, Tong and Xiao, Tianjun and Zhou, Yicong and Zhang, Zheng},
  journal = {Advances in Neural Information Processing Systems},
  year = {2024},
}
```


## Acknowledgement

This working is done during author's internship at Amazon. This codebase is a reimplementation of the work on paper. 
