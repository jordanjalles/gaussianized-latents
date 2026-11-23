from pathlib import Path

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import STL10
import time

data_root = Path(os.environ["ML_DATASETS"])

train_ds = STL10(
    root=data_root,
    split="train+unlabeled",
    download=False, #already downloaded
    transform=transforms.ToTensor(),
)

test_ds = STL10(
    root=data_root,
    split="test",
    download=False, #already downloaded
    transform=transforms.ToTensor(),
)

train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=64 , drop_last=True)
test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=8000)

#hyperparameters
emb_dim = 96 # MHA and MLP width
emb_dim_discriminator = 16
num_blocks = 4 #dinov2 12
n_heads = 6
head_size = 16 #must be divisible by 4 for 2D rope
               #emb_dim = n_heads × head_dim <---------default ViT ratio
n_latent_tokens = 32
lr = 4e-4
layer_scale_gamma = 1e-4 #at init

#acts on x_hat so can be bigger
laplacian_weight = 0.5 

#acts on z so magnitude must be smaller
swd_weight = 0.0002
bwd_weight = 0.0002


TRACK_Z_DIAGNOSTICS = False


patch_size = 12
batches_per_epoch = len(iter(train_dataloader))
b, c, h, w = next(iter(train_dataloader))[0].shape
pc = w//patch_size #patch columns
pr = h//patch_size #patch rows
patch_rows = pr
patch_cols = pc
total_patches = pc * pr
latent_dim = n_latent_tokens*emb_dim
patch_dim = patch_size * c

def apply_2d_rope(q, k, n_latent_tokens, rope):
    assert q.shape == k.shape, 'q and k shapes must match'

    B, H, T, D = q.shape

    assert D % 4 == 0, 'head size must be divisible by 4'

    row_cos, row_sin, col_cos, col_sin = rope
    n_patches = row_cos.shape[0]

    assert T == n_latent_tokens + n_patches

    #match mixed-precision q/k
    row_cos = row_cos.to(dtype=q.dtype)
    row_sin = row_sin.to(dtype=q.dtype)
    col_cos = col_cos.to(dtype=q.dtype)
    col_sin = col_sin.to(dtype=q.dtype)

    def rotate_pairs(x, cos, sin):
        # (B, H, P, D/2)
        x = x.reshape(B, H, n_patches, D//4, 2)

        a, b = x.unbind(dim=-1)

        a_rot = a*cos - b*sin
        b_rot = a*sin + b*cos

        return torch.stack((a_rot, b_rot), dim=-1).flatten(-2)

    def rotate_tensor(x):
        original_T = x.shape[2]

        #latent tokens remain untouched
        x_latent = x[:, :, :n_latent_tokens]
        x_patch = x[:, :, n_latent_tokens:]

        assert x_latent.shape[2] == n_latent_tokens
        assert x_patch.shape[2] == n_patches

        #first half of channels use row position
        #second half of channesl use col position
        x_row, x_col = x_patch.chunk(2,dim=-1)

        x_row = rotate_pairs(x_row, row_cos, row_sin)
        x_col = rotate_pairs(x_col, col_cos, col_sin)

        x_patch = torch.cat((x_row, x_col), dim=-1) #pair up the row and col tokens along D

        x = torch.cat((x_latent, x_patch), dim=2) #stack latent and patch tokens along T
        assert x.shape[2] == original_T, (
            f"RoPE changed token count: {original_T} -> {x.shape[2]}"
        )

        return x

    return rotate_tensor(q), rotate_tensor(k)

#pre-norm multi-head attention
class MHA(nn.Module):
    def __init__(self, 
                 n_heads, 
                 head_size, 
                 emb_dim, 
                 n_latent_tokens
                ):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.emb_dim = emb_dim
        self.n_latent_tokens = n_latent_tokens

        self.norm = nn.LayerNorm(emb_dim)
        self.QKV = nn.Linear(emb_dim, 3*n_heads * head_size)

        self.out_proj = nn.Linear(n_heads * head_size, emb_dim)

        self.gamma = nn.Parameter(torch.full( (emb_dim,), layer_scale_gamma))

    def forward(self, x, rope):
        residual = x
        
        
        #useful for shape viewing/permuting below
        B, T, C = x.shape
        nh = self.n_heads
        hs = self.head_size

        #normalize
        x = self.norm(x)
        
        qkv = self.QKV(x) # (B, T, C) -> ( B, T, 3 * n_heads *head_size )

        #slice qkv aprt
        q = qkv[:, :, : nh * hs ] # ( B, T, n_heads * head_size )
        k = qkv[:, :, nh * hs : 2 * nh * hs ] # ( B, T, n_heads * head_size )
        v = qkv[:, :, 2 * nh * hs :] # ( B, T, n_heads * head_size )


        
        #reshape (B, T, n_heads*headsize) -> (B, n_heads, T, headsize)
        q = q.view(B, T, nh, hs).permute(0, 2, 1, 3)
        k = k.view(B, T, nh, hs).permute(0, 2, 1, 3)
        v = v.view(B, T, nh, hs).permute(0, 2, 1, 3)

        q, k = apply_2d_rope(
            q,
            k,
            n_latent_tokens=self.n_latent_tokens,
            rope=rope,
        )

        #q @ k.t / head_size **0.5
        wei = q @ k.transpose(-1, -2) / (hs**0.5) # (B, H, qT, kT)

        wei = wei.softmax(dim = -1) # each key dimension should total up to 1

        out = wei@v # (qT, kT) @ (vT, headsize) -> (B, H, T, headsize)

        out = out.permute(0, 2, 1, 3).flatten(-2) # (B, T, H*hs)

        out = residual + self.gamma * self.out_proj(out) # (B T C)


        return out
        
        
class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.full((emb_dim,), layer_scale_gamma))
        
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim*4),
            nn.GELU(),
            nn.Linear(emb_dim*4, emb_dim)
        )

    def forward(self, x):
        return x + self.gamma * self.net(x)

#transformer+MLP block
class Block(nn.Module):
    def __init__(self, 
                 n_heads, 
                 head_size, 
                 emb_dim,
                 n_latent_tokens):
        super().__init__()

        self.MHA = MHA(n_heads, head_size, emb_dim, n_latent_tokens)
        self.MLP = MLP(emb_dim)

    def forward(self, x, rope):

        x = self.MHA(x, rope)
        
        x = self.MLP(x)

        return x
        
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, c, emb_dim):
        super().__init__()

        self.patch_size = patch_size

        self.embed = nn.Conv2d(in_channels = c,
                             out_channels = emb_dim,
                             kernel_size = patch_size,
                             stride = patch_size,
                             padding=0)


    def forward(self, x):
        B, C, H, W = x.shape
        patch_rows = H//patch_size
        patch_cols = W//patch_size

        
        out = self.embed(x) # (B, C, H, W) -> (B, emb_dim, pr, pc)
        out = out.flatten(-2).permute(0, 2, 1) # (B, T, C)

        return out, patch_rows, patch_cols

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size, c, emb_dim):
        super().__init__()

        self.patch_size = patch_size
        self.c = c
        self.emb_dim = emb_dim

        self.to_pixels = nn.Linear(emb_dim, c * patch_size**2)

    def forward(self, x, patch_rows, patch_cols):
        B, T, C = x.shape
        p = self.patch_size
        c = self.c

        x = self.to_pixels(x) # BTC -> B, T, c*p**2

        # Split token position and patch contents
        # (B, pr, pc, c, patch, patch)

        x = x.reshape(B,  patch_rows, patch_cols, c, p, p)

        # Interleave patch-grid and within-patch coordinates
        # (B, c, pr, patch, pc, patch)
        x = x.permute(0, 3, 1, 4, 2, 5)

        # Join them into image height and width
        # (B, c, pr*p, pc*p) 
        out = x.reshape(B, c, patch_rows*p, patch_cols*p)

        return out


#define core transformer model 

class CoreTransformer(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 num_blocks, 
                 n_heads, 
                 head_size, 
                 n_latent_tokens
                ):
        
        super().__init__()

        self.n_latent_tokens = n_latent_tokens

        self.blocks = nn.ModuleList([
            Block(n_heads, head_size, emb_dim, n_latent_tokens)
                for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(emb_dim)

        #ROPE BUFFER
        pair_idx = torch.arange(head_size//4, dtype=torch.float32)
        inv_freq = 10_000 ** (-pair_idx / (head_size)//4)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("row_sin_cache", torch.empty(0), persistent=False)
        self.register_buffer("row_cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("col_sin_cache", torch.empty(0), persistent=False)
        self.register_buffer("col_cos_cache", torch.empty(0), persistent=False)

        self.rope_cache_grid = None

    def forward(self, x, patch_rows, patch_cols):
        rope = self.get_rope_cache(patch_rows, patch_cols)
        
        for block in self.blocks:
            x = block(x, rope)

        x = self.norm(x)
    
        return x

    def get_rope_cache(self, patch_rows, patch_cols):
        grid = (patch_rows, patch_cols) #current layout

        if self.rope_cache_grid != grid: #only rebuild when the grid changes
            row_pos = torch.arange(
                patch_rows,
                device = self.inv_freq.device,
            ).repeat_interleave(patch_cols) # rpt interleaved makes pattern e.g. 000111222333 - 000N X cols - 1234...N up to rows

            
            col_pos = torch.arange(
                patch_cols,
                device=self.inv_freq.device,
            ).repeat(patch_rows) # rpt (no interleave) makes pattern e.g. 012301230123 - 0123.0123 X rows - 1234...N up to cols
    
            row_angles = row_pos[:, None] * self.inv_freq[None, :] 
            col_angles = col_pos[:, None] * self.inv_freq[None, :]
    
            self.row_cos_cache = row_angles.cos()
            self.row_sin_cache = row_angles.sin()
            self.col_cos_cache = col_angles.cos()
            self.col_sin_cache = col_angles.sin()
    
            self.rope_cache_grid = grid
        return (
            self.row_cos_cache,
            self.row_sin_cache,
            self.col_cos_cache,
            self.col_sin_cache,
        )

#define encoder model 
from typing import NamedTuple, Optional


class EncoderOutput(NamedTuple):
    z: torch.Tensor
    noise_tokens: Optional[torch.Tensor]
    patch_rows: int
    patch_cols: int
    
class Encoder(nn.Module):
    def __init__(self, 
                 patch_size, 
                 in_channels, 
                 emb_dim, 
                 num_blocks, 
                 n_heads, 
                 head_size, 
                 n_latent_tokens
                ):
        
        super().__init__()

        self.n_latent_tokens = n_latent_tokens
        self.emb_dim = emb_dim

        self.out_norm = nn.LayerNorm(emb_dim*n_latent_tokens, elementwise_affine=False) #reconsider affine False w/ z norm

        self.latent_tokens = nn.Parameter(torch.randn((1, n_latent_tokens, emb_dim)))

        self.mask_token = nn.Parameter(torch.randn((1, 1, emb_dim)))

        
        self.embed = PatchEmbed(patch_size, in_channels, emb_dim)

        self.core_transformer = CoreTransformer(emb_dim = emb_dim, 
                                                 num_blocks = num_blocks, 
                                                 n_heads = n_heads, 
                                                 head_size = head_size, 
                                                 n_latent_tokens = n_latent_tokens
                                                )
    def forward(self, image, mask=None):
        
        patches, patch_rows, patch_cols = self.embed(image)

        raw_patches = patches

        if mask is not None:
            #mask patches
            #mask (B, T)
            #mask token (1, 1, C)
            #patches (B, T, C)
            patches = torch.where(
                mask.unsqueeze(-1),
                self.mask_token,
                patches,
            )
        
        # cat latent tokens with patch embed
        B = patches.shape[0]
        
        latents = self.latent_tokens.expand(B, -1, -1)
        x = torch.cat([latents, patches], dim=1)
        
        x = self.core_transformer(x, patch_rows, patch_cols)

        #slice latent tokens out of x
        latents = x[:, :self.n_latent_tokens]

        patches = x[:, self.n_latent_tokens:]

        #removing out norm in favor of SIGReg loss
        #latents = self.out_norm(latents.flatten(start_dim=1)).reshape_as(latents)

        return EncoderOutput(
            latents, 
            patches, 
            patch_rows, 
            patch_cols
        )

#define decoder model 

class Decoder(nn.Module):
    def __init__(self, 
                 patch_size, 
                 out_channels, 
                 emb_dim, 
                 num_blocks, 
                 n_heads, 
                 head_size, 
                 n_latent_tokens
                ):
        
        super().__init__()

        self.n_latent_tokens = n_latent_tokens

        self.learned_patch_token = nn.Parameter(torch.randn((1,1,emb_dim)))

        self.latent_slots = nn.Parameter(
            torch.randn(1, n_latent_tokens, emb_dim) * 0.02
        )
        
        self.un_embed = PatchUnEmbed(patch_size = patch_size, 
                                     c = out_channels, 
                                     emb_dim = emb_dim)

        self.core_transformer = CoreTransformer(emb_dim = emb_dim, 
                                                 num_blocks = num_blocks, 
                                                 n_heads = n_heads, 
                                                 head_size = head_size, 
                                                 n_latent_tokens = n_latent_tokens
                                                )

        
    def forward(self, enc):
        latents = enc.z + self.latent_slots
        patch_rows = enc.patch_rows
        patch_cols = enc.patch_cols


        #x( B, n_latent_tokens, emb_dim)
        B, T, C = latents.shape

        #copy from (emb_dim) to (B(broadcast), patches*patches(broadcast), emb_dim)
        patches = self.learned_patch_token.expand(B, patch_rows*patch_cols, -1)  
        
        x = torch.cat([latents, patches], dim=1) # cat along token axis

        x = self.core_transformer(x, patch_rows, patch_cols)

        
        #slice patch tokens out of x
        patch_embeddings = x[:, self.n_latent_tokens:]
 
        images = torch.sigmoid(self.un_embed(patch_embeddings, patch_rows, patch_cols))


        return images
# --------------------Def Init Models (and optimizers) --------------------


def init_ae_models():
    
    encoder = Encoder(
        patch_size,
        c,
        emb_dim,
        num_blocks,
        n_heads,
        head_size,
        n_latent_tokens,
    )
    
    decoder = Decoder(
        patch_size,
        c,
        emb_dim,
        num_blocks,
        n_heads,
        head_size,
        n_latent_tokens,
    )
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder.to(device)
    decoder.to(device)
    
    
    # -------------------- Optimizers --------------------
    
    ae_parameters = (
        list(encoder.parameters())
        + list(decoder.parameters())
    )
    n_ae_parameters  = sum(p.numel() for p in ae_parameters)
    print(f"{n_ae_parameters:,} ae parameters")
    
    ae_optimizer = torch.optim.AdamW(
        ae_parameters,
        lr=lr,
    )

    return (
        encoder,
        decoder,
        ae_optimizer,
        device,
    )
