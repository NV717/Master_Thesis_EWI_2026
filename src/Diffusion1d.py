from __future__ import annotations
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import shutil
from pathlib import Path
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from src.DataSets import *
from src.DataSplitting import *
from src.DataPreparation import add_appliance_shares
import numpy as np
import pandas as pd
from ema_pytorch import EMA

from dataclasses import dataclass, fields


'''The general Layout was based on the DDPM Pytorch implementation from https://huggingface.co/blog/annotated-diffusion
and https://github.com/lucidrains/denoising-diffusion-pytorch. Some helper functions and the noise schedule were adopted as is while the Rest
was heavily modified for 1d timeseries use'''

#helper functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# def extract(a, t, x_shape):
#     batch_size = t.shape[0]
#     out = a.gather(-1, t.cpu())
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Downsample1d(dim, dim_out=None):
    return nn.Conv1d(
        dim,
        default(dim_out, dim),
        kernel_size=4,
        stride=2,
        padding=1
    )

def Upsample1d(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim,  default(dim_out, dim), kernel_size=3, padding=1)
    )


#position Embedding

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class StaticConditionEmbedder(nn.Module):
    def __init__(self, cat_cardinalities, n_cont, n_bool, cond_dim, out_dim):
        super().__init__()
        # one Embedding per categorical column (buildingType, buildingAgeBin, ...)
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(card + 1, cond_dim) for card in cat_cardinalities  # +1 for the OrdinalEncoder's unknown/pad index
        ])
        in_dim = len(cat_cardinalities) * cond_dim + n_cont + n_bool
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, static_cat, static_cont, static_bool):
        cat_parts = [emb(static_cat[:, i]) for i, emb in enumerate(self.cat_embeds)]
        c = torch.cat(cat_parts + [static_cont, static_bool], dim=-1)
        return self.mlp(c)

# Model Blocks
class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self,x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 ", "mean")
        var = reduce(weight, "o ... -> o 1 1 ", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block1d(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock1d(nn.Module):
    def __init__(self,dim,dim_out,*,time_emb_dim=None,groups=8):
        super().__init__()

        self.mlp = (
            nn.Sequential(nn.SiLU(),nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block1d(dim,dim_out,groups=groups)

        self.block2 = Block1d(dim_out,dim_out,groups=groups)

        self.res_conv = (nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity())

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)

            time_emb = rearrange(time_emb,"b c -> b c 1")

            scale_shift = time_emb.chunk(2,dim=1)

        h = self.block1(x,scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention1d(nn.Module):
    def __init__(self,dim,heads=4,dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim,hidden_dim * 3,1,bias=False)

        self.to_out = nn.Conv1d(hidden_dim,dim,1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3,dim=1)

        q, k, v = map(lambda t: rearrange(t,"b (h d) n -> b h d n",h=self.heads),qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j",q,k)

        sim = sim - sim.amax(dim=-1,keepdim=True).detach()

        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d",attn,v)

        out = rearrange(out,"b h n d -> b (h d) n")
        return self.to_out(out)

class LinearAttention1D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        #b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class PreNorm1d(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()

        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(
            self.norm(x)
        )


# unet

class Unet1D(nn.Module):
    def __init__(
        self,
        dim=32,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=8,
        cat_cardinalities=None,
        n_cont=0,
        n_bool=0,
        n_dyn_con_cols = 0,

    ):
        super().__init__()

        self.channels = channels
        self.n_dyn_con_cols = n_dyn_con_cols

        self.init_conv = nn.Conv1d(channels +2 + n_dyn_con_cols, dim,kernel_size=3, padding=1) #+2 because extra sin/cos

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock1d,groups=resnet_block_groups) #partial constructor for resnet block

        time_dim = dim * 4 #design convention: widening of conditioning dims

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_embedder = StaticConditionEmbedder(
        cat_cardinalities=cat_cardinalities,
        n_cont=n_cont,
        n_bool=n_bool,
        cond_dim=16, #todo make constructor argument
        out_dim=time_dim,
    )

        # down path
        self.downs = nn.ModuleList([])

        for dim_in, dim_out in in_out:
            self.downs.append(nn.ModuleList([block_klass(dim_in,dim_in,time_emb_dim=time_dim), Downsample1d(dim_in,dim_out)]))

        # Bottleneck
        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim,mid_dim,time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm1d(mid_dim,Attention1d(mid_dim)))
        self.mid_block2 = block_klass(mid_dim,mid_dim,time_emb_dim=time_dim)


        # Up path
        self.ups = nn.ModuleList([])
        for dim_in, dim_out in reversed(in_out):

            self.ups.append(nn.ModuleList([Upsample1d(dim_out,dim_in),block_klass(dim_in * 2,dim_in,time_emb_dim=time_dim)])) #dim_in * 2 cause cat with skip cons


        # Final output
        self.final_block = block_klass(dim,dim,time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim,channels,kernel_size=1)

    def forward(self, x, time, static_cat, static_cont, static_bool, dyn=None):

        b, c, length = x.shape

        positions = torch.arange(length,device=x.device,dtype=x.dtype)

        time_sin = torch.sin(2 * math.pi * positions / 1440) #todo check for problem on different res
        time_cos = torch.cos(2 * math.pi * positions / 1440)

        time_sin = time_sin[None, None, :].expand(b, 1, length)
        time_cos = time_cos[None, None, :].expand(b, 1, length)

        parts= [x]
        if self.n_dyn_con_cols >0:
            parts.append(dyn)
        parts += [time_sin, time_cos]    # Todo put time directily in x
        x = torch.cat(parts,dim=1)

        x = self.init_conv(x)

        t = self.time_mlp(time) + self.cond_embedder(static_cat, static_cont, static_bool)

        skips = []
        # DownPath
        for block, downsample in self.downs:

            x = block(x, t)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Up path
        for upsample, block in self.ups:
            x = upsample(x)
            skip = skips.pop()

            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")

            x = torch.cat((x, skip),dim=1)
            x = block(x, t)

        # Final
        x = self.final_block(x, t)
        return self.final_conv(x)


#noise schedules
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


BETA_SCHEDULES = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
    "quadratic": quadratic_beta_schedule,
    "sigmoid": sigmoid_beta_schedule,
}
@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    @property
    def timesteps(self) -> int:
        return len(self.betas)

    @classmethod
    def from_betas(cls, betas: torch.Tensor) -> DiffusionSchedule:
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)
        return cls(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=ac,
            alphas_cumprod_prev=ac_prev,
            sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
            sqrt_alphas_cumprod=torch.sqrt(ac),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - ac),
            sqrt_recip_alphas_cumprod=torch.sqrt(1.0 / ac),
            sqrt_recipm1_alphas_cumprod=torch.sqrt(1.0 / ac - 1.0),
            posterior_variance=betas * (1.0 - ac_prev) / (1.0 - ac),
            posterior_mean_coef1=betas * torch.sqrt(ac_prev) / (1.0 - ac),
            posterior_mean_coef2=(1.0 - ac_prev) * torch.sqrt(alphas) / (1.0 - ac),
        )

    @classmethod
    def make(cls, name: str = "sigmoid", timesteps: int = 1000) -> DiffusionSchedule:
        return cls.from_betas(BETA_SCHEDULES[name](timesteps))

    def to(self, device) -> DiffusionSchedule:
        return type(self)(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def extract(self, name: str, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = getattr(self, name).gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    # Forward Diffusion Process
    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return (self.extract("sqrt_alphas_cumprod", t, x0.shape) * x0
                + self.extract("sqrt_one_minus_alphas_cumprod", t, x0.shape) * noise)


#Loss function

def p_losses(denoise_model, schedule,x_start, t, static_cat, static_cont, static_bool, noise=None, dyn=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = schedule.q_sample(x0=x_start, t=t, noise=noise)

    drop_mask = torch.rand(x_start.shape[0], device=x_start.device) < 0.0
    static_cat = static_cat.clone()
    static_cont = static_cont.clone()
    static_bool = static_bool.clone()
    static_cat[drop_mask] = 0          # index 0 ie unknown
    static_cont[drop_mask] = 0.0
    static_bool[drop_mask] = 0.0


    predicted_noise = denoise_model(x_noisy, t, static_cat, static_cont, static_bool, dyn) # ToDO adapt for CFG

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# sampling

@torch.inference_mode()
def compute_channel_minmax(dataloader, n_channels, key="target", channel_dim=1, progress=True):
    reduce_dims = tuple(d for d in range(3) if d != channel_dim)
    lo = torch.full((n_channels,), float("inf"))
    hi = torch.full((n_channels,), float("-inf"))

    for batch in tqdm(dataloader, disable=not progress):
        x = batch[key].float().cpu()
        lo = torch.minimum(lo, x.amin(dim=reduce_dims))
        hi = torch.maximum(hi, x.amax(dim=reduce_dims))

    return lo.view(1, -1, 1), hi.view(1, -1, 1)

@torch.inference_mode()
def p_sample(model, x, t, t_index,schedule,static_cat, static_cont, static_bool, dyn=None,clip_range=True):
    pred_noise = model(x, t, static_cat, static_cont, static_bool, dyn)

    # predict x0 from x_t and the predicted noise
    x0 = (schedule.extract("sqrt_recip_alphas_cumprod", t, x.shape) * x
          - schedule.extract("sqrt_recipm1_alphas_cumprod", t, x.shape) * pred_noise)
    if clip_range is not None:
        lo, hi = clip_range
        x0 = torch.clamp(x0, lo.to(x.device), hi.to(x.device))

    # posterior mean q(x_{t-1} | x_t, x0)
    model_mean = (schedule.extract("posterior_mean_coef1", t, x.shape) * x0
                  + schedule.extract("posterior_mean_coef2", t, x.shape) * x)

    if t_index == 0:
        return model_mean
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(schedule.extract("posterior_variance", t, x.shape)) * noise

@torch.inference_mode()
def p_sample_loop(model, shape, schedule, static_cat, static_cont, static_bool,
                  dyn=None, clip_range=None, return_trajectory=False):
    device = next(model.parameters()).device
    schedule = schedule.to(device)
    b = shape[0]
    x = torch.randn(shape, device=device)
    traj = []

    for i in tqdm(reversed(range(schedule.timesteps)), desc="sampling", total=schedule.timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, i, schedule, static_cat, static_cont, static_bool,
                     dyn, clip_range)
        if return_trajectory:
            traj.append(x.cpu().numpy())
    return traj if return_trajectory else x.cpu()


@torch.inference_mode()
def sample1d(model, sequence_length, batch_size, channels, schedule, static_cat,
             static_cont, static_bool, dyn=None, clip_range=None, return_trajectory=False):
    return p_sample_loop(
        model,
        shape=(batch_size, channels, sequence_length),
        schedule=schedule,
        static_cat=static_cat,
        static_cont=static_cont,
        static_bool=static_bool,
        dyn= dyn,
        clip_range =clip_range,
        return_trajectory=return_trajectory)


@torch.inference_mode()
def sample_year(model,schedule, dyn_days, static_cat, static_cont, static_bool, K, n_tgt, channels, length, reverse,clip_range=None,):
    # dyn_days: list of (B, n_dyn, L) tensors, one per day (occupancy, Off Day, ...)
    prev, out = None, []
    for dyn in dyn_days:
        b = dyn.shape[0]
        if K > 0:
            ctx = torch.zeros(b, n_tgt + 1, length, device=dyn.device)
            if prev is not None:
                tail = prev[:, :, -K:]
                if reverse:
                    tail = tail.flip(-1)                       # newest minute in slot 0
                ctx[:, :n_tgt, :K] = tail.to(dyn.device)
                ctx[:, n_tgt, :K] = 1.0
            dyn = torch.cat([dyn, ctx], dim=1)
        prev = sample1d(model, length, b, channels, schedule, static_cat, static_cont,static_bool, dyn=dyn, clip_range=clip_range)
        out.append(prev)
    return torch.cat(out, dim=-1)



