import numpy as np
import torch
from .diffusion import SDTraining
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from regional_attention.region_reorganization import prepare_train_attention_kwargs, prepare_empty_attention_kwargs

class GLIGENSDTraining(SDTraining):
    def __init__(
            self, 
            *args, 
            train_image_size=512, 
            test_image_size=512, 
            accumulate_grad_batches=8,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.train_image_size = train_image_size
        self.test_image_size = test_image_size
        self.max_objs = 40

        self.unet.enable_gradient_checkpointing()

        self.set_gradient_to_false(self.text_encoder)
        self.set_gradient_to_false(self.vae)

        for name, param in self.unet.named_parameters():
            if 'position_net' in name or 'fuser' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.automatic_optimization = False
        self.acc_grad_batches = accumulate_grad_batches

        if self.use_ema:
            self.init_ema_model()

        self.scheduler.set_timesteps(self.num_ddim_steps)

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

    def prepare_attn_kwargs_batch(
        self, 
        batch, 
        device, 
        dtype
    ):
        bboxes = batch['bboxes']
        labels = batch['labels']
        res = []
        for l, b in zip(labels, bboxes):
            res.append(
                self.prepare_train_attn_kwargs(
                    b, l, device, dtype
                )
            )
        return {
            k: torch.cat([r[k] for r in res], dim=0) for k in res[0].keys()
        }

    def prepare_train_attn_kwargs(
        self,
        bboxes, labels,
        device, dtype
    ):
        gligen_phrases = labels
        gligen_boxes = bboxes
        max_objs = 30
        if len(gligen_boxes) > max_objs:
            gligen_phrases = gligen_phrases[:max_objs]
            gligen_boxes = gligen_boxes[:max_objs]
        # prepare batched input to the GLIGENTextBoundingboxProjection (boxes, phrases, mask)
        # Get tokens for phrases from pre-trained CLIPTokenizer
        tokenizer_inputs = self.tokenizer(
            gligen_phrases, 
            padding=True, 
            return_tensors="pt",
            truncation=True,
            max_length=77
        ).to(device)
        # For the token, we use the same pre-trained text encoder
        # to obtain its text feature
        _text_embeddings = self.text_encoder(**tokenizer_inputs).pooler_output
        n_objs = len(gligen_boxes)
        # For each entity, described in phrases, is denoted with a bounding box,
        # we represent the location information as (xmin,ymin,xmax,ymax)
        boxes = torch.zeros(max_objs, 4, device=device, dtype=dtype)
        boxes[:n_objs] = torch.tensor(gligen_boxes)
        text_embeddings = torch.zeros(
            max_objs, self.unet.config.cross_attention_dim, device=device, dtype=dtype
        )
        text_embeddings[:n_objs] = _text_embeddings
        # Generate a mask for each object that is entity described by phrases
        masks = torch.zeros(max_objs, device=device, dtype=dtype)
        masks[:n_objs] = 1

        repeat_batch = 1
        boxes = boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        text_embeddings = text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        masks = masks.unsqueeze(0).expand(repeat_batch, -1).clone()
        return {"boxes": boxes, "positive_embeddings": text_embeddings, "masks": masks}

    def process_batch(self, batch, mode):
        assert mode in ['train', 'val', 'test']
        image = batch['image']
        text = batch['caption']
        image = image.to(self.vae.dtype)
        image = self.encode_image_to_latent(image)
        b, *_ = image.shape
        noise = torch.randn_like(image)
        if mode == 'train':
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=image.device).long()
            x_t = self.add_noise(image, t, noise=noise)
        else:
            t = torch.full((b,), self.scheduler.config.num_train_timesteps-1, device=image.device, dtype=torch.long)
            x_t = self.add_noise(image, t, noise=noise)

        prompt_embeds = self.encode_text(text)
        rnd_number = np.random.rand()
        if rnd_number < 0.1:
            prompt_embeds = torch.zeros_like(prompt_embeds)

        model_kwargs = {
            'encoder_hidden_states': prompt_embeds,
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


    def training_step(self, batch, batch_idx):
        N = self.acc_grad_batches

        res_dict = super().training_step(batch, batch_idx)
        loss = res_dict['loss'] / N

        opt = self.optimizers()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')

        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad()
        return res_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        text = batch['caption'][0]
        bboxes = batch['bboxes'][0] # exclude batch size dim
        labels = batch['labels'][0]
        res = self.pipe(
            prompt=text,
            height=self.test_image_size, width=self.test_image_size,
            gligen_boxes=bboxes / self.test_image_size, 
            gligen_phrases=labels,
            gligen_scheduled_sampling_beta=1, 
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=self.guidance_scale,
        )
        return np.array(res[0][0])[None] / 255. # 1, H, W, 3

    def configure_optimizers(self):
        import bitsandbytes as bnb
        params_to_train = []
        for name, param in self.unet.named_parameters():
            if 'position_net' in name or 'fuser' in name:
                params_to_train.append(param)
        optimizer = bnb.optim.AdamW8bit(params_to_train, **self.optim_args)
        return optimizer

    