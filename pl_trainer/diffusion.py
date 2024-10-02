import torch
from torch import nn
import pytorch_lightning as pl
from misc_utils.model_utils import default, instantiate_from_config
from einops import rearrange, repeat
from diffusers import StableDiffusionPipeline

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class DDPM(pl.LightningModule):
    def __init__(
        self, 
        pipe: StableDiffusionPipeline, # this should be a diffuser pipe
        loss_fn='l2',
        optim_args={},
        **kwargs
    ):
        '''
        denoising_fn: a denoising model such as UNet
        beta_schedule_args: a dictionary which contains
            the configurations of the beta schedule
        '''
        super().__init__(**kwargs)
        self.pipe = pipe
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.vae = pipe.vae
        if hasattr(pipe, 'tokenizer'):
            self.tokenizer = pipe.tokenizer
        if hasattr(pipe, 'text_encoder'):
            self.text_encoder = pipe.text_encoder
        if hasattr(pipe, 'text_encoder_2') and hasattr(pipe, 'tokenizer_2'):
            # sdxl
            self.tokenizer_2 = pipe.tokenizer_2
            self.text_encoder_2 = pipe.text_encoder_2

        # stable video diffusion
        if hasattr(pipe, 'image_encoder'):
            self.image_encoder = pipe.image_encoder
        if hasattr(pipe, 'feature_extractor'):
            self.feature_extractor = pipe.feature_extractor

        self.optim_args = optim_args
        self.loss = loss_fn
        if loss_fn == 'l2' or loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_fn == 'l1' or loss_fn == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif isinstance(loss_fn, dict):
            self.loss_fn = instantiate_from_config(loss_fn)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def add_noise(self, x, t, noise=None):
        noise = default(noise, torch.randn_like(x))
        return self.scheduler.add_noise(x, noise, t)

    def predict_x_0_from_x_t(self, model_output: torch.Tensor, t: torch.LongTensor, x_t: torch.Tensor):
        ''' recover x_0 from predicted noise. Reverse of Eq(4) in DDPM paper
        \hat(x_0) = 1 / sqrt[\bar(a)]*x_t - sqrt[(1-\bar(a)) / \bar(a)]*noise'''
        return torch.cat(
            [self.scheduler.step(
                model_output[i:i+1], int(t[i]), x_t[i:i+1]
            ).pred_original_sample for i in range(len(t))],
            dim=0
        ) # DDIMScheduler's step requires timestep to be int
        # alphas_cumprod = self.scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)
        # sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod[t]).flatten()
        # sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod[t] - 1.).flatten()
        # while len(sqrt_recip_alphas_cumprod.shape) < len(x_t.shape):
        #     sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.unsqueeze(-1)
        #     sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod.unsqueeze(-1)
        # return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * model_output

    def predict_x_tm1_from_x_t(self, model_output, t, x_t):
        '''predict x_{t-1} from x_t and model_output'''
        return self.scheduler.step(model_output, t, x_t).prev_sample

class SDTraining(DDPM):
    def __init__(
        self, 
        pipe: StableDiffusionPipeline, # this should be a diffuser pipe
        loss_fn='l2',
        optim_args={
            'lr': 1e-4,
            'weight_decay': 5e-4
        },
        log_args={}, # for record all arguments with self.save_hyperparameters
        num_ddim_steps=20,
        guidance_scale=5.,
        use_ema=False,
        ema_decay=0.99,
        ema_start=10000,
        **kwargs
    ):
        super().__init__(
            pipe=pipe,
            loss_fn=loss_fn, 
            optim_args=optim_args,
            **kwargs)
        self.log_args = log_args
        self.call_save_hyperparameters()

        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_start = ema_start

    def init_ema_model(self):
        if self.use_ema:
            raise NotImplementedError('init_ema_model should be implemented in the inherit class if use_ema is True')

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['unet', 'vae', 'tokenizer', 'text_encoder', 'tokenizer_2', 'text_encoder_2', 'pipe', 'scheduler', 'image_encoder', 'feature_extractor'])

    def process_batch(self, batch, mode):
        assert mode in ['train', 'val', 'test']
        image, text = batch['image'], batch['text']
        image = self.encode_image_to_latent(image)
        b, *_ = image.shape
        noise = torch.randn_like(image)
        if mode == 'train':
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=image.device).long()
            x_t = self.add_noise(image, t, noise=noise)
        else:
            t = torch.full((b,), self.scheduler.config.num_train_timesteps-1, device=image.device, dtype=torch.long)
            x_t = self.add_noise(image, t, noise=noise)


        model_kwargs = {
            'encoder_hidden_states': self.encode_text(text),
            'cross_attention_kwargs': None
        }
        '''the order of return is 
            1) model input, 
            2) model pred target, 
            3) model time condition
            4) raw image before adding noise
            5) model_kwargs
        '''
        return {
            'model_input': x_t,
            'model_target': noise,
            't': t,
            'model_kwargs': model_kwargs
        }

    @torch.no_grad()
    def encode_text(self, prompt):
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt, device=self.text_encoder.device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )
        return prompt_embeds

    @torch.no_grad()
    def encode_image_to_latent(self, x):
        x = (x - 0.5) * 2
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent_to_image(self, x, return_type='torch'):
        x = x / self.vae.config.scaling_factor
        img = self.vae.decode(x, return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)
        if return_type == 'torch':
            return img
        elif return_type == 'numpy':
            return img.permute(0, 2, 3, 1).float().detach().cpu().numpy()
        else:
            raise NotImplementedError('return_type should be either torch or numpy')

    def get_loss(self, pred, target, t):
        loss_raw = self.loss_fn(pred, target)
        loss_flat = mean_flat(loss_raw)

        loss = loss_flat
        loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        processed_batch = self.process_batch(batch, mode='train')
        x_t = processed_batch['model_input']
        y = processed_batch['model_target']
        t = processed_batch['t']
        model_kwargs = processed_batch['model_kwargs']
        pred = self.unet(
            x_t, 
            t, 
            return_dict=False,
            **model_kwargs
        )[0]
        loss = self.get_loss(pred, y, t)
        x_0_hat = self.predict_x_0_from_x_t(pred, t, x_t)
        x_0_hat = self.decode_latent_to_image(x_0_hat)

        self.log(f'train_loss', loss, sync_dist=True)
        return {
            'loss': loss,
            'model_input': x_t,
            'model_output': pred,
            'x_0_hat': x_0_hat
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sampled = self.pipe(
            prompt=batch['text'],
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=self.guidance_scale,
            guidance_rescale=0.7,
            output_type='np'
        ).images
        return sampled

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), **self.optim_args)
        return optimizer

class SDXLTraining(SDTraining):
    def __init__(self, *args, train_image_size=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_image_size = train_image_size

    def process_batch(self, batch, mode):
        assert mode in ['train', 'val', 'test']
        image, text = batch['image'], batch['text']
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
        # here is a tricky part, what if my output size changes during inference
        add_time_ids = self.pipe._get_add_time_ids(
            original_size=(self.train_image_size, self.train_image_size), 
            crops_coords_top_left=(0, 0),
            target_size=(self.train_image_size, self.train_image_size),
            dtype=prompt_embeds.dtype
        ).to(prompt_embeds.device)
        model_kwargs = {
            'encoder_hidden_states': prompt_embeds,
            'added_cond_kwargs': {
                'text_embeds': add_text_embeds,
                'time_ids': add_time_ids
            },
            'cross_attention_kwargs': None
        }
        '''the order of return is 
            1) model input, 
            2) model pred target, 
            3) model time condition
            4) raw image before adding noise
            5) model_kwargs
        '''
        return {
            'model_input': x_t,
            'model_target': noise,
            't': t,
            'model_kwargs': model_kwargs
        }

    @torch.no_grad()
    def encode_text(self, prompt):
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt, device=self.text_encoder.device, num_images_per_prompt=1
        )
        return prompt_embeds, pooled_prompt_embeds
