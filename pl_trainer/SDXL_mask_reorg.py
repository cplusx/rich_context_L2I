import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from .diffusion import SDXLTraining
from misc_utils.model_utils import get_obj_from_str
from einops import repeat
from regional_attention.region_reorganization import prepare_train_attention_kwargs, prepare_empty_attention_kwargs

class RegionalAttnSDXLTraining(SDXLTraining):
    def __init__(
            self, 
            *args, 
            train_image_size=512, 
            test_image_size=1024, 
            lora_rank=None,
            position_net_and_fuser_init_weights=None,
            accumulate_grad_batches=8,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.train_image_size = train_image_size
        self.test_image_size = test_image_size
        self.max_objs = 40

        self.unet.enable_gradient_checkpointing()

        self.set_gradient_to_false(self.text_encoder)
        self.set_gradient_to_false(self.text_encoder_2)
        self.set_gradient_to_false(self.vae)

        if lora_rank is not None and lora_rank > 0:
            from peft import LoraConfig
            self.lora_rank = lora_rank
            unet_lora_config = LoraConfig(
                r = lora_rank, 
                lora_alpha = lora_rank,
                init_lora_weights='gaussian',
                target_modules=["to_k", "to_q", "to_v", "to_out.0"]
            )
            self.unet.add_adapter(unet_lora_config)
            print('INFO: LoRA initialized with rank', lora_rank)

        for name, param in self.unet.named_parameters():
            if 'position_net' in name or 'fuser' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if position_net_and_fuser_init_weights is not None:
            print('INFO: loading position net and fuser weights')
            pn_and_fuser_weights = torch.load(position_net_and_fuser_init_weights)
            unet_sd = self.unet.state_dict()
            for k, v in pn_and_fuser_weights.items():
                if k in unet_sd and unet_sd[k].shape == v.shape:
                    unet_sd[k] = v
            self.unet.load_state_dict(unet_sd)

        self.automatic_optimization = False
        self.acc_grad_batches = accumulate_grad_batches

        if self.use_ema:
            self.init_ema_model()

    def init_ema_model(self):
        if self.ema_decay:
            self.ema_unet = AveragedModel(self.unet, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
            if self.local_rank == 0:
                print('INFO: EMA model enabled with decay', self.ema_decay)
    
    def set_gradient_to_false(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['unet', 'vae', 'tokenizer', 'text_encoder', 'tokenizer_2', 'text_encoder_2', 'pipe', 'scheduler', 'image_encoder', 'feature_extractor'])

    def training_step(self, batch, batch_idx):
        N = self.acc_grad_batches
        opt = self.optimizers()

        res_dict = super().training_step(batch, batch_idx)
        loss = res_dict['loss'] / N

        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')

        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad()

            if self.use_ema and self.global_step > self.ema_start:
                if self.local_rank == 0:
                    print(f'INFO: updating EMA model @ step {self.global_step}')
                self.ema_unet.update_parameters(self.unet)
        
        return res_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError('validation_step not implemented')

    def configure_optimizers(self):
        import bitsandbytes as bnb
        params_to_train = []
        for name, param in self.unet.named_parameters():
            if 'position_net' in name or 'fuser' in name or 'lora' in name:
                params_to_train.append(param)
        optimizer = bnb.optim.AdamW8bit(params_to_train, **self.optim_args)
        return optimizer

class RegionalAttnSDXLTrainingMaskAndBboxReorgAdaptiveText(RegionalAttnSDXLTraining):
    def __init__(self, *args, attention_weight_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weight_scale = attention_weight_scale

    def prepare_attn_kwargs_batch(
        self, 
        batch,
        device, 
        dtype
    ):
        reorg_bboxes = batch['reorg_bboxes']
        reorg_labels = batch['reorg_labels']
        reorg_masks = batch['reorg_masks']
        reorg_union_masks = batch['reorg_union_masks']
        reorg_ious = batch['reorg_ious']
        sep_token = batch['sep_token']
        res = []
        for l, m, um, b, iou in zip(reorg_labels, reorg_masks, reorg_union_masks, reorg_bboxes, reorg_ious):
            rnd_number = np.random.rand()
            if rnd_number > 0.1:
                res.append(
                    prepare_train_attention_kwargs(
                        reorg_bboxes=b,
                        reorg_labels=l,
                        reorg_masks=m,
                        reorg_union_masks=um,
                        reorg_ious=iou,
                        sep_token=sep_token,
                        cross_attention_dim=self.unet.config.cross_attention_dim,
                        tokenizer=self.tokenizer,
                        tokenizer_2=self.tokenizer_2,
                        text_encoder=self.text_encoder,
                        text_encoder_2=self.text_encoder_2,
                        device=device,
                        dtype=dtype,
                        attention_weight_scale=self.attention_weight_scale
                    )
                )
            else:
                res.append(
                    prepare_empty_attention_kwargs(
                        mask=m,
                        cross_attention_dim=self.unet.config.cross_attention_dim,
                        tokenizer=self.tokenizer,
                        text_encoder=self.text_encoder,
                        device=device,
                        dtype=dtype,
                    )
                )
        return {
            k: torch.cat([r[k] for r in res], dim=0) for k in res[0].keys()
        }

    def process_batch(self, batch, mode):
        assert mode in ['train', 'val', 'test']
        image = batch['image']
        text = batch['caption']
        image = self.encode_image_to_latent(image)
        b, *_ = image.shape
        noise = torch.randn_like(image)
        if mode == 'train':
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=image.device).long()
            x_t = self.add_noise(image, t, noise=noise)
        else:
            t = torch.full((b,), self.scheduler.config.num_train_timesteps-1, device=image.device, dtype=torch.long)
            x_t = self.add_noise(image, t, noise=noise)

        prompt_embeds, pooled_prompt_embeds = self.encode_text(text)
        add_text_embeds = pooled_prompt_embeds.to(prompt_embeds.device)
        rnd_number = np.random.rand()
        if rnd_number < 0.1:
            prompt_embeds = torch.zeros_like(prompt_embeds)

        add_time_ids = self.pipe._get_add_time_ids(
            original_size=(self.train_image_size, self.train_image_size), 
            crops_coords_top_left=(0, 0),
            target_size=(self.train_image_size, self.train_image_size),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        ).to(prompt_embeds.device)
        add_time_ids = repeat(add_time_ids, '() d -> b d', b=b)

        model_kwargs = {
            'encoder_hidden_states': prompt_embeds,
            'added_cond_kwargs': {
                'text_embeds': add_text_embeds,
                'time_ids': add_time_ids
            },
            'cross_attention_kwargs': {
                'gligen': self.prepare_attn_kwargs_batch(
                    batch,
                    prompt_embeds.device, 
                    image.dtype)
            }
        }
        return {
            'model_input': x_t,
            'model_target': noise,
            't': t,
            'model_kwargs': model_kwargs,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        text = batch['caption'][0]
        bboxes = batch['bboxes'][0] # exclude batch size dim
        labels = batch['labels'][0]
        res = self.pipe(
            prompt=text,
            height=self.test_image_size, width=self.test_image_size,
            boxes=bboxes, # mask reorg requires input to be int (unnormalized)
            labels=labels,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=self.guidance_scale,
            attention_weight_scale=self.attention_weight_scale
        )
        return np.array(res[0][0])[None] / 255. # 1, H, W, 3