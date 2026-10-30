import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from src.Diffusion1d import exists, default, Residual,SinusoidalPositionEmbeddings, StaticConditionEmbedder,DiffusionSchedule, BETA_SCHEDULES,p_losses, predict_noise, p_sample, p_sample_loop, sample_dpm, sample1d, sample_year,compute_channel_minmax


'''The general Layout was based on the DDPM Pytorch implementation from https://huggingface.co/blog/annotated-diffusion
and https://github.com/lucidrains/denoising-diffusion-pytorch. Some helper functions and the noise schedule were adopted as is while the Rest
was heavily modified for 1d timeseries use'''



def Upsample2d(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample2d(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)



class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self,x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block2d(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
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


class ResnetBlock2d(nn.Module):
    def __init__(self,dim,dim_out,*,time_emb_dim=None,groups=8):
        super().__init__()

        self.mlp = (
            nn.Sequential(nn.SiLU(),nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block2d(dim,dim_out,groups=groups)

        self.block2 = Block2d(dim_out,dim_out,groups=groups)

        self.res_conv = (nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity())

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)

            time_emb = rearrange(time_emb,"b c -> b c 1 1")

            scale_shift = time_emb.chunk(2,dim=1)

        h = self.block1(x,scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention2d(nn.Module):
    def __init__(self,dim,heads=4,dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim,hidden_dim * 3,1,bias=False)
        self.to_out = nn.Conv2d(hidden_dim,dim,1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3,dim=1)

        q, k, v = map(lambda t: rearrange(t,"b (h c) x y -> b h c (x y)",h=self.heads),qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j",q,k)

        sim = sim - sim.amax(dim=-1,keepdim=True).detach()

        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d",attn,v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm2d(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(self.norm(x))


# unet

class Unet2D(nn.Module):
    def __init__(
        self,
        dim=32,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=8,
        cat_cardinalities=None,
        n_cont=0,
        n_bool=0,
        n_dyn_cond_cols = 0,
        n_days = 1,
        steps_per_day = 1440,

    ):
        super().__init__()

        self.channels = channels
        self.n_dyn_cond_cols = n_dyn_cond_cols

        self.n_days = n_days
        self.steps_per_days = steps_per_day

        self.init_conv = nn.Conv2d(channels +2 + n_dyn_cond_cols, dim,kernel_size=3, padding=1) #+2 because extra sin/cos

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock2d,groups=resnet_block_groups) #partial constructor for resnet block

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
            self.downs.append(nn.ModuleList([block_klass(dim_in,dim_in,time_emb_dim=time_dim), Downsample2d(dim_in,dim_out)]))

        # Bottleneck
        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim,mid_dim,time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm2d(mid_dim,Attention2d(mid_dim)))
        self.mid_block2 = block_klass(mid_dim,mid_dim,time_emb_dim=time_dim)


        # Up path
        self.ups = nn.ModuleList([])
        for dim_in, dim_out in reversed(in_out):

            self.ups.append(nn.ModuleList([Upsample2d(dim_out,dim_in),block_klass(dim_in * 2,dim_in,time_emb_dim=time_dim)])) #dim_in * 2 cause cat with skip cons


        # Final output
        self.final_block = block_klass(dim,dim,time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim,channels,kernel_size=1)

    def forward(self, x, time, static_cat, static_cont, static_bool, dyn=None):

        b, c, length = x.shape
        D, S = self.n_days, self.steps_per_da
        x = x.view(b, c, D, S)

        positions = torch.arange(length,device=x.device,dtype=x.dtype)

        tod_sin = torch.sin(2 * math.pi * positions / S).view(1, 1, 1, S).expand(b, 1, D, S)
        tod_cos = torch.cos(2 * math.pi * positions / S).view(1, 1, 1, S).expand(b, 1, D, S)

        parts= [x]
        if self.n_dyn_cond_cols >0:
            parts.append(dyn)
        parts += [tod_sin, tod_cos]
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

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

            x = torch.cat((x, skip),dim=1)
            x = block(x, t)

        x = self.final_block(x, t)
        return self.final_conv(x).reshape(b, self.channels, L)






