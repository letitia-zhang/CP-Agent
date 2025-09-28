""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer,\
    text_global_pool
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    in_chans: int = 3 

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 256
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
            in_chans=vision_cfg.in_chans,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            in_chans=vision_cfg.in_chans,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


    
class TrunkNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):

        for i, layer in enumerate(self.net):
            x = layer(x)
        
        return x




class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        tokenizer=None,
        special_tokens: Optional[List[str]] = None,
    ):
        super().__init__()
        self.output_dict = output_dict
        
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)


        self.special_tokens = special_tokens
        self.tokenizer = tokenizer
        
        # Logit scale和bias
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None
        
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd = {'positional_embedding'}
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        return no_wd
    
    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_text(
        self, 
        text, 
        normalize: bool = False,
        concentration: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        compound_embedding: Optional[torch.Tensor] = None
    ):
        cast_dtype = self.transformer.get_cast_dtype()

        # Handle tokenization if text is string (for compatibility)
        if isinstance(text, (list, str)):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required when input text is string")
            texts = [text] if isinstance(text, str) else text
            tokens = self.tokenizer(texts, context_length=self.context_length)
            text = tokens
        
        
        # Get token embeddings
        x = self.token_embedding(text).to(cast_dtype)
        # Continue with standard processing
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = text_global_pool(x, text, self.text_pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        # if compound_embedding is not None and concentration is not None and time is not None:
        #     compound_features, concentration_features, time_features = self.multi_trunk(compound_embedding, concentration, time)
        #     x = torch.cat([x, compound_features, concentration_features, time_features], dim=-1)

        if normalize:
            x = F.normalize(x, dim=-1)

        return x

    def get_logits(
        self, 
        image, 
        text, 
        concentration: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        compound_embedding: Optional[torch.Tensor] = None
    ):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(
            text, 
            normalize=True,
            concentration=concentration,
            time=time,
            compound_embedding=compound_embedding
        )
        
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        
        return image_logits, text_logits
    


    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert image is not None and text is not None, 'Both image and text inputs are required to compute logits'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if text is not None:
            cast_dtype = self.transformer.get_cast_dtype()
            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.to(cast_dtype)
            x, intermediates = self.transformer.forward_intermediates(
                x,
                attn_mask=self.attn_mask,
                indices=text_indices
            )
            if normalize_intermediates:
                intermediates = [self.ln_final(xi) for xi in intermediates]

            # NOTE this model doesn't support cls embed in text transformer, no need for extra intermediate tokens
            output["text_intermediates"] = intermediates

            if not intermediates_only:
                x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
                x = text_global_pool(x, text, self.text_pool_type)
                if self.text_projection is not None:
                    if isinstance(self.text_projection, nn.Linear):
                        x = self.text_projection(x)
                    else:
                        x = x @ self.text_projection
                if normalize:
                    x = F.normalize(x, dim=-1)
                output["text_features"] = x

        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None

        if output_logits:
            image_logits = logit_scale_exp * output["image_features"] @ output["text_features"].T
            if self.logit_bias is not None:
                image_logits += self.logit_bias
            text_logits = image_logits.T
            output["image_logits"] = image_logits
            output["text_logits"] = text_logits

        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output


    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        concentration: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        compound_embedding: Optional[torch.Tensor] = None,
    ):
        
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) 
        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()



class EnhancedCLIP(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        tokenizer=None,
        special_tokens: Optional[List[str]] = None,
        compound_dim: int = 174,
    ):
        super().__init__()
        self.output_dict = output_dict
        
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)


        hidden_dim = max(embed_dim, 512)
        self.conc_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.compound_mlp = nn.Sequential(
            nn.Linear(compound_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 设置特殊token
        if special_tokens is None:
            special_tokens = ['<CONC_TOKEN>', '<TIME_TOKEN>', '<COMPOUND_TOKEN>']
        self.special_tokens = special_tokens
        
        # 存储tokenizer
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'special_token_ids'):
            self.special_token_ids = {
                tok: tokenizer.special_token_ids[tok]
                for tok in special_tokens
            }
        else:
            raise ValueError("Tokenizer must support special_token_ids()")

        print(f"[Auto] Fetched special token IDs from tokenizer: {self.special_token_ids}")

        
        # Logit scale和bias
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None
    
    def _add_special_tokens(self):
        """Add special tokens to tokenizer"""
        print("🧩 Adding special tokens to tokenizer...")

        if hasattr(self.tokenizer, 'encoder'):  # SimpleTokenizer
            vocab_size = len(self.tokenizer.encoder)
            for i, token in enumerate(self.special_tokens):
                if token not in self.tokenizer.encoder:
                    self.tokenizer.encoder[token] = vocab_size + i
                    self.tokenizer.decoder[vocab_size + i] = token
                    print(f"Registered special token: {token} -> ID {vocab_size + i}")
                else:
                    print(f"Token already exists: {token} -> ID {self.tokenizer.encoder[token]}")

            # self.tokenizer.vocab_size = len(self.tokenizer.encoder)
            for token in self.special_tokens:
                self.tokenizer.cache[token] = token

            import regex as re
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.tokenizer.pat = re.compile(
                f"(?:{special_pattern})|'s|'t|'re|'ve|'m|'ll|'d|[\p{{L}}]+|[\p{{N}}]+|[^\s\p{{L}}\p{{N}}]+",
                re.IGNORECASE,
            )

            print(f"Tokenizer regex updated for special tokens: {special_pattern}")
            # print(f"Final vocab size: {self.tokenizer.vocab_size}")

            self.special_token_ids = {
                token: self.tokenizer.encoder[token] for token in self.special_tokens 
                if token in self.tokenizer.encoder
            }
            print(f"Special token IDs: {self.special_token_ids}")
        
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd = {'positional_embedding'}
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        return no_wd
    
    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_text(
        self, 
        text, 
        normalize: bool = False,
        concentration: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        compound_embedding: Optional[torch.Tensor] = None
    ):
        cast_dtype = self.transformer.get_cast_dtype()

        # Handle tokenization if text is string (for compatibility)
        if isinstance(text, (list, str)):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required when input text is string")
            
            # Convert to list if single string
            texts = [text] if isinstance(text, str) else text
            
            # Tokenize text
            tokens = self.tokenizer(texts, context_length=self.context_length)
            text = tokens
        
        # If text is already tokenized (from data.py), use it directly
        # text should be a tensor of token ids
        
        # Get token embeddings
        x = self.token_embedding(text).to(cast_dtype)
        
        # Replace special tokens with numerical data if available
        if hasattr(self, 'special_token_ids') and (concentration is not None or time is not None or compound_embedding is not None):
            x = self._replace_special_tokens(x, text, concentration, time, compound_embedding, cast_dtype)
        
        # Continue with standard processing
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = text_global_pool(x, text, self.text_pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        # if compound_embedding is not None and concentration is not None and time is not None:
        #     compound_features, concentration_features, time_features = self.multi_trunk(compound_embedding, concentration, time)
        #     x = torch.cat([x, compound_features, concentration_features, time_features], dim=-1)

        if normalize:
            x = F.normalize(x, dim=-1)

        return x

    def _replace_special_tokens(
            self, 
            token_embeddings: torch.Tensor, 
            input_ids: torch.Tensor, 
            concentration: Optional[torch.Tensor], 
            time: Optional[torch.Tensor], 
            compound_embedding: Optional[torch.Tensor], 
            cast_dtype: torch.dtype
        ) -> torch.Tensor:
            """Replace manually assigned special token IDs with MLP-mapped embeddings"""
            batch_size, seq_len, embed_dim = token_embeddings.shape

            # # Create MLPs only once
            # if not hasattr(self, 'conc_mlp'):
            #     hidden_dim = max(embed_dim, 512)
            #     self.conc_mlp = nn.Sequential(
            #         nn.Linear(2, hidden_dim),
            #         nn.GELU(),
            #         nn.Linear(hidden_dim, embed_dim)
            #     ).to(token_embeddings.device).to(cast_dtype)

            #     self.time_mlp = nn.Sequential(
            #         nn.Linear(1, hidden_dim),
            #         nn.GELU(),
            #         nn.Linear(hidden_dim, embed_dim)
            #     ).to(token_embeddings.device).to(cast_dtype)

            #     self.compound_mlp = nn.Sequential(
            #         nn.Linear(174, hidden_dim),
            #         nn.GELU(),
            #         nn.Linear(hidden_dim, embed_dim)
            #     ).to(token_embeddings.device).to(cast_dtype)

            # Replace <CONC_TOKEN>
            if concentration is not None and '<CONC_TOKEN>' in self.special_token_ids:
                conc_token_id = self.special_token_ids['<CONC_TOKEN>']
                conc_mask = (input_ids == conc_token_id)

                if conc_mask.any():
                    conc_embeddings = self.conc_mlp(concentration.to(cast_dtype))  # [B, D]
                    conc_positions = conc_mask.nonzero(as_tuple=True)
                    if conc_positions[0].numel() > 0:
                        batch_indices = conc_positions[0]
                        seq_indices = conc_positions[1]
                        token_embeddings[batch_indices, seq_indices] = conc_embeddings[batch_indices]

            # Replace <TIME_TOKEN>
            if time is not None and '<TIME_TOKEN>' in self.special_token_ids:
                time_token_id = self.special_token_ids['<TIME_TOKEN>']
                time_mask = (input_ids == time_token_id)
                if time_mask.any():
                    if time.dim() == 1:
                        time = time.unsqueeze(-1)
                    time_embeddings = self.time_mlp(time.to(cast_dtype))  # [B, D]
                    time_positions = time_mask.nonzero(as_tuple=True)
                    if time_positions[0].numel() > 0:
                        batch_indices = time_positions[0]
                        seq_indices = time_positions[1]
                        token_embeddings[batch_indices, seq_indices] = time_embeddings[batch_indices]

            # Replace <COMPOUND_TOKEN>
            if compound_embedding is not None and '<COMPOUND_TOKEN>' in self.special_token_ids:
                compound_token_id = self.special_token_ids['<COMPOUND_TOKEN>']
                compound_mask = (input_ids == compound_token_id)
                if compound_mask.any():
                    compound_embeddings = self.compound_mlp(compound_embedding.to(cast_dtype))  # [B, D]
                    compound_positions = compound_mask.nonzero(as_tuple=True)
                    if compound_positions[0].numel() > 0:
                        batch_indices = compound_positions[0]
                        seq_indices = compound_positions[1]
                        token_embeddings[batch_indices, seq_indices] = compound_embeddings[batch_indices]

            return token_embeddings
    
    def get_logits(
        self, 
        image, 
        text, 
        concentration: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        compound_embedding: Optional[torch.Tensor] = None
    ):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(
            text, 
            normalize=True,
            concentration=concentration,
            time=time,
            compound_embedding=compound_embedding
        )
        
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        
        return image_logits, text_logits
    


    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            image: Input image tensor
            text: Input text tensor
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            text_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply final norm layer to all intermediates
            normalize: L2 Normalize final features
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            text_output_fmt: Shape of intermediate text feature outputs (ignored for this model)
            text_output_extra_tokens: Return both prefix and spatial intermediate tokens (ignored for this model)
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert image is not None and text is not None, 'Both image and text inputs are required to compute logits'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if text is not None:
            cast_dtype = self.transformer.get_cast_dtype()
            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.to(cast_dtype)
            x, intermediates = self.transformer.forward_intermediates(
                x,
                attn_mask=self.attn_mask,
                indices=text_indices
            )
            if normalize_intermediates:
                intermediates = [self.ln_final(xi) for xi in intermediates]

            # NOTE this model doesn't support cls embed in text transformer, no need for extra intermediate tokens
            output["text_intermediates"] = intermediates

            if not intermediates_only:
                x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
                x = text_global_pool(x, text, self.text_pool_type)
                if self.text_projection is not None:
                    if isinstance(self.text_projection, nn.Linear):
                        x = self.text_projection(x)
                    else:
                        x = x @ self.text_projection
                if normalize:
                    x = F.normalize(x, dim=-1)
                output["text_features"] = x

        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None

        if output_logits:
            image_logits = logit_scale_exp * output["image_features"] @ output["text_features"].T
            if self.logit_bias is not None:
                image_logits += self.logit_bias
            text_logits = image_logits.T
            output["image_logits"] = image_logits
            output["text_logits"] = text_logits

        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output


    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        concentration: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        compound_embedding: Optional[torch.Tensor] = None,
    ):
        
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True, concentration=concentration, time=time, compound_embedding=compound_embedding) 
        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat




def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 2, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    example_concentration = torch.rand((batch_size, 2), device=device)
    example_time = torch.rand((batch_size, 1), device=device)

    compound_dim = model.compound_mlp[0].in_features
    example_compound_embedding = torch.rand((batch_size, compound_dim), device=device)
    
    # example_compound_embedding = torch.rand((batch_size, 174), device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text, example_concentration, example_time, example_compound_embedding),
            encode_text=(example_text, True, example_concentration, example_time, example_compound_embedding),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info('Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg