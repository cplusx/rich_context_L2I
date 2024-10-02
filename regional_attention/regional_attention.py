import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_fourier_embeds_from_boundingbox
from diffusers.utils import is_torch_version
from einops import rearrange, repeat

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TextEmbeddingNet(nn.Module):
    def __init__(self, text_feat_dim, out_dim):
        super().__init__()

        self.text_feat_dim = text_feat_dim
        self.out_dim = out_dim

        self.linears = nn.Sequential(
            nn.Linear(self.text_feat_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.text_feat_dim]))

    def forward(
        self,
        object_masks,
        text_embeddings=None, # text embedding
        **kwargs
    ):
        '''
        object_masks: (B, num_objs)
        text embeddings is the text embedding of shape (B, num_objs, 77, text dim)
        '''

        text_null = self.null_text_feature.view(1, 1, 1, -1)
        masks_ = object_masks.unsqueeze(-1).unsqueeze(-1)

        text_embeddings = text_embeddings * masks_ + (1 - masks_) * text_null

        text_feat = self.linears(text_embeddings)

        return {
            'text_feat': text_feat,
            'object_masks': object_masks,
            **kwargs
        }

class TextEmbeddingNetV2(TextEmbeddingNet):
    # V2 add the bounding box coordinates into the text embeddings
    def __init__(self, text_feat_dim, out_dim, fourier_freqs=8):
        super().__init__(text_feat_dim=text_feat_dim, out_dim=out_dim)

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        # overwrite the linears
        self.linears = nn.Sequential(
            nn.Linear(self.text_feat_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.text_feat_dim + self.position_dim]))

    def forward(
        self,
        object_masks,
        bbox_embeddings=None, # bounding box embedding
        text_embeddings=None, # text embedding
        **kwargs
    ):
        '''
        object_masks: (B, num_objs)
        Positive embeddings is the text embedding of shape (B, num_objs, 77, dim)
        bbox embeddings is the bounding box embedding of shape (B, num_objs, 77, 4)
        '''

        # text embedding
        text_null = self.null_text_feature.view(1, 1, 1, -1)
        masks_ = object_masks.unsqueeze(-1).unsqueeze(-1)


        # bboxes  # B*N*77*4 -> B*N*77*C
        bbox_embeddings = rearrange(bbox_embeddings, 'b n l d -> (b n) l d')
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, bbox_embeddings)
        bbox_embeddings = rearrange(xyxy_embedding, '(b n) l d -> b n l d', b=text_embeddings.shape[0], n=text_embeddings.shape[1])

        text_embeddings = torch.cat([text_embeddings, bbox_embeddings], dim=-1)
        text_embeddings = text_embeddings * masks_ + (1 - masks_) * text_null

        text_feat = self.linears(text_embeddings)

        return {
            'text_feat': text_feat,
            'object_masks': object_masks,
            **kwargs
        }

class RegionalCrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        self.attn = Attention(
            query_dim=query_dim, 
            cross_attention_dim=context_dim, 
            heads=n_heads, 
            dim_head=d_head,
            residual_connection=False
        )
        zero_module(self.attn.to_out[0])
        self.register_parameter('null_text_feat', nn.Parameter(torch.zeros([query_dim, ])))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.enabled:
            return x
        
        x_residual = x

        B, L, dim = x.shape

        if kwargs.get('H', None) and kwargs.get('W', None):
            # if input is not square, we need to pass the original H and W
            ori_H, ori_W = kwargs['H'], kwargs['W']
            downscale = ori_H * ori_W / L
            H = int(ori_H / downscale ** 0.5)
            W = int(ori_W / downscale ** 0.5)
        else:
            # by default, we assume the input is square
            H = W = int(L ** 0.5)
            if H * W != L:
                raise ValueError(f"Input is not square, Got L={L}, but H*W={H*W}")

        reorg_masks = objs['reorg_masks'] # B, N, H, W
        # resize the reorg masks to the same size as the input
        reorg_masks = F.interpolate(reorg_masks.float(), size=(H, W), mode='nearest')
        reorg_text_feat = objs['text_feat'] # B, N, 77, dim
        num_objs = reorg_masks.shape[1]

        x = repeat(x, 'b l d -> b n l d', n=num_objs)
        x_flat = rearrange(x, 'b n l d -> (b n) l d')
        text_feat_flat = rearrange(reorg_text_feat, 'b n l d -> (b n) l d')

        text_feat_flat = text_feat_flat.to(x_flat.dtype)

        if 'cross_attention_mask' in objs:
            # NOTE: this value should already have been scaled to -inf to 0.
            cross_attn_mask = objs['cross_attention_mask']
            cross_attn_mask = rearrange(cross_attn_mask, 'b n l -> (b n) l').to(x.dtype)
        else:
            cross_attn_mask = None

        x_obj = self.attn(x_flat, text_feat_flat, attention_mask=cross_attn_mask)
        # x_obj = self.attn(x_flat, text_feat_flat)
        x_obj = rearrange(x_obj, '(b n) l d -> b n l d', b=B, n=num_objs)

        region_masks = torch.where(reorg_masks > 0.5, 1., 0.).to(x.dtype)
        region_masks = rearrange(region_masks, 'b n h w -> b n (h w) ()')
        x = (x_obj * region_masks).sum(dim=1)

        x_null = repeat(self.null_text_feat, 'd -> b l d', b=B, l=L)
        x = x + (1 - region_masks.sum(dim=1)) * x_null

        return x + x_residual

class RegionalCrossAndSelfAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        self.attn = Attention(
            query_dim=query_dim, 
            cross_attention_dim=context_dim, 
            heads=n_heads, 
            dim_head=d_head,
            residual_connection=False,
        )
        zero_module(self.attn.to_out[0])
        self.register_parameter('null_text_feat', nn.Parameter(torch.zeros([query_dim, ])))

        self.self_attn = Attention(
            query_dim=query_dim,
            heads=n_heads,
            dim_head=d_head,
            residual_connection=False,
        )
        zero_module(self.self_attn.to_out[0])

        self.enabled = True
        self.gradient_checkpointing = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor, **kwargs) -> torch.Tensor:

        if not self.enabled:
            return x
        
        # 1. Layout Cross Attention
        x_residual = x

        B, L, dim = x.shape

        if kwargs.get('H', None) and kwargs.get('W', None):
            # if input is not square, we need to pass the original H and W
            ori_H, ori_W = kwargs['H'], kwargs['W']
            downscale = ori_H * ori_W / L
            H = int(ori_H / downscale ** 0.5)
            W = int(ori_W / downscale ** 0.5)
        else:
            # by default, we assume the input is square
            H = W = int(L ** 0.5)
            if H * W != L:
                raise ValueError(f"Input is not square, Got L={L}, but H*W={H*W}")

        reorg_intersection_masks = objs['reorg_masks'] # B, N, H, W
        # resize the reorg masks to the same size as the input
        reorg_intersection_masks = F.interpolate(reorg_intersection_masks.float(), size=(H, W), mode='nearest')
        reorg_text_feat = objs['text_feat'] # B, N, 77, dim
        num_objs = reorg_intersection_masks.shape[1]

        x = repeat(x, 'b l d -> b n l d', n=num_objs)
        x_flat = rearrange(x, 'b n l d -> (b n) l d')
        text_feat_flat = rearrange(reorg_text_feat, 'b n l d -> (b n) l d')

        text_feat_flat = text_feat_flat.to(x_flat.dtype)

        if 'cross_attention_mask' in objs:
            # NOTE: this value should already have been scaled to -inf to 0.
            cross_attn_mask = objs['cross_attention_mask']
            cross_attn_mask = rearrange(cross_attn_mask, 'b n l -> (b n) l').to(x.dtype)
        else:
            cross_attn_mask = None
        x_obj = self.attn(x_flat, text_feat_flat, attention_mask=cross_attn_mask)
        x_obj = rearrange(x_obj, '(b n) l d -> b n l d', b=B, n=num_objs)

        region_masks = torch.where(reorg_intersection_masks > 0.5, 1., 0.).to(x.dtype)
        region_masks = rearrange(region_masks, 'b n h w -> b n (h w) ()')
        x = (x_obj * region_masks).sum(dim=1)

        x_null = repeat(self.null_text_feat, 'd -> b l d', b=B, l=L)
        x = x + (1 - region_masks.sum(dim=1)) * x_null

        x = x + x_residual

        # 2. Regional Self Attention
        x_residual = x

        reorg_union_masks = objs['reorg_union_masks']
        reorg_union_masks = F.interpolate(reorg_union_masks.float(), size=(H, W), mode='nearest')
        attn_mask = rearrange(reorg_union_masks, 'b n h w -> (b n) (h w)').to(x.dtype)
        attn_mask = (1 - attn_mask) * -1e3 # change to -inf for masked_fill

        x = repeat(x, 'b l d -> b n l d', n=num_objs)
        x_flat = rearrange(x, 'b n l d -> (b n) l d')
        # x = self.self_attn(x_flat, attention_mask=attn_mask)
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attn),
                x_flat,
                None,
                attn_mask,
                **ckpt_kwargs
            )
        else:
            x = self.self_attn(x_flat, attention_mask=attn_mask)

        x = rearrange(x, '(b n) l d -> b n l d', b=B, n=num_objs)

        x = (x * region_masks).sum(dim=1)

        x = x + x_residual
        return x

class InstDiffStyleSelfAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        self.attn = Attention(
            query_dim=query_dim, 
            cross_attention_dim=context_dim, 
            heads=n_heads, 
            dim_head=d_head,
            residual_connection=False,
        )
        zero_module(self.attn.to_out[0])
        self.register_parameter('null_text_feat', nn.Parameter(torch.zeros([query_dim, ])))

        self.self_attn = Attention(
            query_dim=query_dim,
            heads=n_heads,
            dim_head=d_head,
            residual_connection=False,
        )
        zero_module(self.self_attn.to_out[0])

        self.enabled = True
        self.gradient_checkpointing = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor, **kwargs) -> torch.Tensor:

        if not self.enabled:
            return x
        
        # 1. Layout Cross Attention
        x_residual = x

        B, L, dim = x.shape

        if kwargs.get('H', None) and kwargs.get('W', None):
            # if input is not square, we need to pass the original H and W
            ori_H, ori_W = kwargs['H'], kwargs['W']
            downscale = ori_H * ori_W / L
            H = int(ori_H / downscale ** 0.5)
            W = int(ori_W / downscale ** 0.5)
        else:
            # by default, we assume the input is square
            H = W = int(L ** 0.5)
            if H * W != L:
                raise ValueError(f"Input is not square, Got L={L}, but H*W={H*W}")

        reorg_intersection_masks = objs['reorg_masks'] # B, N, H, W
        # resize the reorg masks to the same size as the input
        reorg_intersection_masks = F.interpolate(reorg_intersection_masks.float(), size=(H, W), mode='nearest')
        reorg_text_feat = objs['text_feat'] # B, N, 77, dim
        num_objs = reorg_intersection_masks.shape[1]

        x = repeat(x, 'b l d -> b n l d', n=num_objs)
        x_flat = rearrange(x, 'b n l d -> (b n) l d')
        text_feat_flat = rearrange(reorg_text_feat, 'b n l d -> (b n) l d')

        text_feat_flat = text_feat_flat.to(x_flat.dtype)

        if 'cross_attention_mask' in objs:
            # NOTE: this value should already have been scaled to -inf to 0.
            cross_attn_mask = objs['cross_attention_mask']
            cross_attn_mask = rearrange(cross_attn_mask, 'b n l -> (b n) l').to(x.dtype)
        else:
            cross_attn_mask = None
        x_obj = self.attn(x_flat, text_feat_flat, attention_mask=cross_attn_mask)
        x_obj = rearrange(x_obj, '(b n) l d -> b n l d', b=B, n=num_objs)

        region_masks = torch.where(reorg_intersection_masks > 0.5, 1., 0.).to(x.dtype)
        region_masks = rearrange(region_masks, 'b n h w -> b n (h w) ()')
        x = (x_obj * region_masks).sum(dim=1)

        x_null = repeat(self.null_text_feat, 'd -> b l d', b=B, l=L)
        x = x + (1 - region_masks.sum(dim=1)) * x_null

        x = x + x_residual

        # 2. Regional Self Attention
        x_residual = x

        reorg_union_masks = objs['reorg_union_masks']
        reorg_union_masks = F.interpolate(reorg_union_masks.float(), size=(H, W), mode='nearest')
        attn_mask = rearrange(reorg_union_masks, 'b n h w -> (b n) (h w)').to(x.dtype)
        attn_mask = (1 - attn_mask) * -1e3 # change to -inf for masked_fill

        x = repeat(x, 'b l d -> b n l d', n=num_objs)
        x_flat = rearrange(x, 'b n l d -> (b n) l d')
        # x = self.self_attn(x_flat, attention_mask=attn_mask)
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attn),
                x_flat,
                None,
                attn_mask,
                **ckpt_kwargs
            )
        else:
            x = self.self_attn(x_flat, attention_mask=attn_mask)

        x = rearrange(x, '(b n) l d -> b n l d', b=B, n=num_objs)

        x = (x * region_masks).sum(dim=1)

        x = x + x_residual
        return x

# for instance diffusion ablation, can be removed

class TextEmbeddingNetInstDiff(TextEmbeddingNet):
    # V2 add the bounding box coordinates into the text embeddings
    def __init__(self, text_feat_dim, out_dim, fourier_freqs=8):
        super().__init__(text_feat_dim=text_feat_dim, out_dim=out_dim)

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        # overwrite the linears
        self.linears = nn.Sequential(
            nn.Linear(self.text_feat_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.text_feat_dim + self.position_dim]))

    def forward(
        self,
        object_masks,
        bbox_embeddings=None, # bounding box embedding
        text_embeddings=None, # text embedding
        **kwargs
    ):
        '''
        object_masks: (B, num_objs)
        Positive embeddings is the text embedding of shape (B, num_objs, 77, dim)
        bbox embeddings is the bounding box embedding of shape (B, num_objs, 77, 4)
        '''

        # text embedding
        text_null = self.null_text_feature.view(1, 1, 1, -1)
        masks_ = object_masks.unsqueeze(-1).unsqueeze(-1)


        # bboxes  # B*N*77*4 -> B*N*1*C
        bbox_embeddings = rearrange(bbox_embeddings, 'b n l d -> (b n) l d')[:, :1]
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, bbox_embeddings)
        bbox_embeddings = rearrange(xyxy_embedding, '(b n) l d -> b n l d', b=text_embeddings.shape[0], n=text_embeddings.shape[1])

        text_embeddings = torch.cat([text_embeddings, bbox_embeddings], dim=-1)
        text_embeddings = text_embeddings * masks_ + (1 - masks_) * text_null

        text_feat = self.linears(text_embeddings)

        return {
            'text_feat': text_feat,
            'object_masks': object_masks,
            **kwargs
        }

class InstDiffSelfAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        self.register_parameter('null_text_feat', nn.Parameter(torch.zeros([context_dim, ])))

        self.text_linear = nn.Sequential(
            nn.Linear(context_dim, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
        )

        self.self_attn = Attention(
            query_dim=query_dim,
            heads=n_heads,
            dim_head=d_head,
            residual_connection=False,
        )
        zero_module(self.self_attn.to_out[0])

        self.enabled = True
        self.gradient_checkpointing = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor, **kwargs) -> torch.Tensor:

        if not self.enabled:
            return x
        
        # 1. prepare self-attention mask for joint visual and textual tokens
        x_residual = x

        B, L, dim = x.shape

        if kwargs.get('H', None) and kwargs.get('W', None):
            # if input is not square, we need to pass the original H and W
            ori_H, ori_W = kwargs['H'], kwargs['W']
            downscale = ori_H * ori_W / L
            H = int(ori_H / downscale ** 0.5)
            W = int(ori_W / downscale ** 0.5)
        else:
            # by default, we assume the input is square
            H = W = int(L ** 0.5)
            if H * W != L:
                raise ValueError(f"Input is not square, Got L={L}, but H*W={H*W}")

        reorg_intersection_masks = objs['reorg_masks'] # B, N, H, W
        # resize the reorg masks to the same size as the input
        reorg_intersection_masks = F.interpolate(reorg_intersection_masks.float(), size=(H, W), mode='nearest')
        reorg_text_feat = objs['text_feat'] # B, N, 1, dim
        num_objs = reorg_intersection_masks.shape[1]

        text_feat_flat = rearrange(reorg_text_feat, 'b n 1 d -> b n d')
        text_feat_flat = text_feat_flat.to(x.dtype)
        text_feat_flat = self.text_linear(text_feat_flat)

        region_masks = torch.where(reorg_intersection_masks > 0.5, 1., 0.).to(x.dtype)
        region_masks = rearrange(region_masks, 'b n h w -> b n (h w) ()')

        # 2. Regional Self Attention
        joint_flat = torch.cat(
            [x, text_feat_flat], dim=1
        ) # b, l, d + b, n, d -> b, n+l, d (l is number of visual tokens, n is number of textual tokens)
        joint_flat = repeat(joint_flat, 'b l d -> b n l d', n=num_objs)
        joint_flat = rearrange(joint_flat, 'b n l d -> (b n) l d')

        reorg_union_masks = objs['reorg_union_masks']
        reorg_union_masks = F.interpolate(reorg_union_masks.float(), size=(H, W), mode='nearest')

        e = torch.eye(num_objs, device=x.device, dtype=x.dtype)
        visual_textual_masks = repeat(1 - e, 'n m -> (b n) m', b=B) * -1e3

        attn_mask = rearrange(reorg_union_masks, 'b n h w -> (b n) (h w)').to(x.dtype)
        attn_mask = (1 - attn_mask) * -1e3 # change to -inf for masked_fill

        attn_mask = torch.cat([attn_mask, visual_textual_masks], dim=1)

        # x = self.self_attn(x_flat, attention_mask=attn_mask)
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attn),
                joint_flat,
                None,
                attn_mask,
                **ckpt_kwargs
            )
        else:
            x = self.self_attn(joint_flat, attention_mask=attn_mask)

        x = x[:, :L, :] # remove the textual tokens
        x = rearrange(x, '(b n) l d -> b n l d', b=B, n=num_objs)

        x = (x * region_masks).sum(dim=1)

        x = x + x_residual
        return x
