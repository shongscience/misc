"""
================================================================================
spherex_anomaly_vit.py
================================================================================

SPHEREx Anomaly Detection using Vision Transformer (ViT)

Authors : Sungryong Hong & Claude AI (Anthropic)
Version : v0.2
Date    : April 2026

--------------------------------------------------------------------------------
Overview
--------------------------------------------------------------------------------
This module provides a complete inference pipeline for detecting anomalous
observations in SPHEREx satellite imagery using a pretrained Vision Transformer
(ViT) binary classifier.

Two input modes are supported:
  - .npy input  : preprocessed 6-band image cubes (6, 2040, 2040)
  - FITS input  : raw SPHEREx Level 2 FITS files (6 bands per observation)

In both cases, the preprocessing pipeline downsamples each band from 2040x2040
to 255x255, computes a quality map from NaN pixels, applies per-band robust
(Median MAD) scaling, and concatenates the results into a 4-channel image
(4, 255, 765) for model inference.

--------------------------------------------------------------------------------
Model Architecture
--------------------------------------------------------------------------------
  VisionTransformerBinaryClassifier
    - Input  : (B, 4, 255, 765)
    - Patches: 15x15, grid = 17x51 = 867 tokens
    - embed_dim=128, hidden_dim=256, num_heads=4, num_layers=4
    - Output : scalar logit per sample -> sigmoid -> anomaly probability

--------------------------------------------------------------------------------
API Summary
--------------------------------------------------------------------------------
Preprocessing (.npy):
    downsize_with_quality(file_path)         -- 2040 -> 255 + quality map
    robust_scale(arr)                        -- per-band Median MAD scaling
    build_4ch(down_img, quality)             -- 6ch x2 -> 4ch concat
    preprocess_file(file_path)               -- full pipeline: .npy -> (4,255,765)

Preprocessing (FITS):
    _get_image_from_fits(fits_path, factor)  -- read & flag single FITS band
    _array_to_quality(arr)                   -- downscaled array -> quality map
    preprocess_fits(band_fits_paths, factor) -- 6 FITS -> (4,255,765)
    parse_fits_dir(fits_dir)                 -- group FITS files by obs_id

Model:
    AttentionBlock                           -- Multi-head self-attention block
    VisionTransformerBinaryClassifier        -- Full ViT classifier
    load_model(ckpt_path)                    -- load pretrained checkpoint

Data:
    SpherexInferenceDataset                  -- Dataset for .npy files
    SpherexFitsDataset                       -- Dataset for FITS files
    build_dataloader(file_list, ...)         -- DataLoader for .npy
    build_fits_dataloader(fits_dir, ...)     -- DataLoader for FITS

Inference:
    run_inference(model, loader, device)           -- silent inference
    run_inference_tqdm(model, loader, device)      -- inference with progress bar (.npy)
    run_inference_fits_tqdm(model, loader, device) -- inference with progress bar (FITS)

Grad-Attention (.npy):
    compute_grad_attention_maps(model, file_path, device, patch_size)
    visualize_grad_attention(file_path, pos_map, neg_map, prob, ...)

Visualization (.npy):
    visualize_raw_bands(file_path, prob, ...)      -- raw 6-band image + histogram
    visualize_model_input(file_path, prob, ...)    -- preprocessed 4-channel input

Visualization (FITS):
    visualize_raw_bands_fits(band_paths, prob, ...) -- raw 6-band FITS + histogram
    visualize_model_input_fits(band_paths, prob, .) -- preprocessed 4-channel input
    visualize_grad_attention_fits(model, band_paths, ...) -- Grad-Attention overlay

--------------------------------------------------------------------------------
Dependencies
--------------------------------------------------------------------------------
  torch, numpy, matplotlib
  astropy       (FITS input only)
  skimage       (FITS input only)
  tqdm          (progress bar inference)
  PIL           (GIF generation)
================================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ==============================================================================
# Preprocessing — .npy input
# ==============================================================================

def downsize_with_quality(file_path):
    """
    Load a raw .npy file and downsample from 2040x2040 to 255x255
    using 8x8 block averaging, while computing a quality map.

    The quality map represents the fraction of valid (non-NaN) pixels
    within each 8x8 block, ranging from 0.0 (all NaN) to 1.0 (all valid).
    Blocks where all pixels are NaN are set to 0.0 (not NaN) to avoid
    downstream RuntimeWarnings during scaling.

    Parameters
    ----------
    file_path : str
        Path to .npy file of shape (6, 2040, 2040), dtype float32.

    Returns
    -------
    down_img : np.ndarray, shape (6, 255, 255), float32
        Downsampled image. All-NaN blocks are set to 0.0.
    quality  : np.ndarray, shape (6, 255, 255), float32
        Valid pixel fraction per 8x8 block. Range [0, 1].
    """
    arr = np.load(file_path)
    C, H, W = arr.shape

    # Reshape into 8x8 blocks: (C, H//8, 8, W//8, 8)
    arr_block = arr.reshape(C, H//8, 8, W//8, 8)
    valid_mask = np.isfinite(arr_block)
    valid_count = valid_mask.sum(axis=(2, 4))          # (C, 255, 255)

    # Replace NaN with 0 before summing to avoid RuntimeWarning
    arr_filled = np.where(valid_mask, arr_block, 0.0)
    block_sum = arr_filled.sum(axis=(2, 4))

    down_img = block_sum / np.maximum(valid_count, 1)  # safe division
    down_img[valid_count == 0] = 0.0                   # all-NaN block -> 0

    quality = valid_count.astype(np.float32) / 64.0    # fraction [0, 1]

    return down_img.astype(np.float32), quality.astype(np.float32)


def robust_scale(arr):
    """
    Apply per-band robust scaling using Median Absolute Deviation (MAD).

    For each band:
      1. Compute median (m) and MAD from finite (non-NaN) pixels only.
      2. Scale: z = (x - m) / (1.4826 * MAD + eps)
      3. Replace NaN pixels with 0.0.
      4. Clip to [-5, 5].

    The factor 1.4826 makes MAD consistent with the standard deviation
    under a Gaussian distribution.

    Parameters
    ----------
    arr : np.ndarray, shape (6, 255, 255)
        Downsampled image from downsize_with_quality().

    Returns
    -------
    out : np.ndarray, shape (6, 255, 255), float32
        Robustly scaled image. NaN pixels become 0.0. Clipped to [-5, 5].
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        band = arr[c]
        finite = band[np.isfinite(band)]

        # All-NaN band: fill with 0.0 (quality map handles masking)
        if len(finite) == 0:
            out[c] = 0.0
            continue

        m = np.median(finite)
        mad = np.median(np.abs(finite - m))
        sigma = 1.4826 * mad + 1e-6                    # avoid division by zero

        scaled = np.where(np.isfinite(band), (band - m) / sigma, 0.0)
        out[c] = np.clip(scaled, -5, 5)

    return out


def build_4ch(down_img, quality):
    """
    Concatenate downsampled bands and quality maps into a 4-channel image
    compatible with the ViT model input.

    The 6 bands are split into two groups (bandA: 1-2-3, bandB: 4-5-6),
    and each group's 3 tiles are concatenated horizontally to form a
    255x765 image. The same is done for quality maps (qualA, qualB).

    Parameters
    ----------
    down_img : np.ndarray, shape (6, 255, 255), float32
        Robustly scaled downsampled image bands.
    quality  : np.ndarray, shape (6, 255, 255), float32
        Quality maps from downsize_with_quality().

    Returns
    -------
    out : np.ndarray, shape (4, 255, 765), float32
        4-channel model input:
          ch 0 (bandA) : bands 1, 2, 3 concatenated horizontally
          ch 1 (bandB) : bands 4, 5, 6 concatenated horizontally
          ch 2 (qualA) : quality maps of bands 1, 2, 3 concatenated horizontally
          ch 3 (qualB) : quality maps of bands 4, 5, 6 concatenated horizontally
    """
    x = np.concatenate([down_img, quality], axis=0)    # (12, 255, 255)

    out = np.empty((4, 255, 255*3), dtype=np.float32)
    out[0] = np.concatenate([x[0],  x[1],  x[2]],  axis=1)  # bandA
    out[1] = np.concatenate([x[3],  x[4],  x[5]],  axis=1)  # bandB
    out[2] = np.concatenate([x[6],  x[7],  x[8]],  axis=1)  # qualA
    out[3] = np.concatenate([x[9],  x[10], x[11]], axis=1)  # qualB

    return out


def preprocess_file(file_path):
    """
    Full preprocessing pipeline for a single .npy file.

    Chains: downsize_with_quality -> robust_scale -> build_4ch

    Parameters
    ----------
    file_path : str
        Path to .npy file of shape (6, 2040, 2040).

    Returns
    -------
    img_4ch : np.ndarray, shape (4, 255, 765), float32
        Model-ready 4-channel input tensor.
    """
    down_img, quality = downsize_with_quality(file_path)
    scaled = robust_scale(down_img)
    return build_4ch(scaled, quality)


# ==============================================================================
# Preprocessing — FITS input
# ==============================================================================

def _get_image_from_fits(fits_path, downscale_factor):
    """
    Read a single SPHEREx Level 2 FITS file and return a preprocessed image.

    Processing steps:
      1. Read IMAGE, FLAGS, and ZODI HDUs.
      2. Identify bad pixels using FLAGS bitmask:
           - Bits 0-2  : SUR flags (TRANSIENT, OVERFLOW, SUR_ERROR)
           - MP_SOURCE : source-contaminated pixels
           - MP_NONFUNC: non-functional detector pixels
      3. Set flagged pixels to NaN.
      4. Subtract zodiacal light (ZODI) background.
      5. Downsample using 8x8 block mean (NaN-aware).

    Parameters
    ----------
    fits_path       : str or Path
        Path to a SPHEREx Level 2 FITS file.
    downscale_factor: int
        Spatial downscaling factor. 8 -> 2040x2040 becomes 255x255.

    Returns
    -------
    image : np.ndarray, shape (H//factor, W//factor), float32
        Preprocessed and downscaled image. NaN where all pixels in a block
        were flagged or invalid.
    """
    from astropy.io import fits as afits

    with afits.open(fits_path) as hdulist:
        image = hdulist["IMAGE"].data.astype(np.float32)
        flags = hdulist["FLAGS"].data
        zodi  = hdulist["ZODI"].data

        # Bitmask: SUR flags (bits 0,1,2) + SOURCE + NONFUNC
        flag_val = (
            7
            + (1 << hdulist["FLAGS"].header["MP_SOURCE"])
            + (1 << hdulist["FLAGS"].header["MP_NONFUNC"])
        )
        has_flag = (flags & flag_val) > 0
        image[has_flag] = np.nan
        image -= zodi

    if downscale_factor == 1:
        return image

    # 8x8 block mean (NaN-aware, no RuntimeWarning)
    H, W = image.shape
    H2 = H // downscale_factor
    W2 = W // downscale_factor

    arr_block = image.reshape(H2, downscale_factor, W2, downscale_factor)
    valid_mask = np.isfinite(arr_block)
    valid_count = valid_mask.sum(axis=(1, 3))

    arr_filled = np.where(valid_mask, arr_block, 0.0)
    block_sum = arr_filled.sum(axis=(1, 3))

    result = block_sum / np.maximum(valid_count, 1)
    result[valid_count == 0] = np.nan                  # all-NaN block stays NaN

    return result.astype(np.float32)


def _array_to_quality(arr):
    """
    Convert a downscaled 6-band array into a quality map and a NaN-free array.

    For FITS input, quality is defined as a binary mask:
      - 1.0 where the pixel is finite (valid)
      - 0.0 where the pixel is NaN (flagged or missing)

    Note: This differs from the .npy quality map, which uses the fraction of
    valid pixels within an 8x8 block. The FITS quality is pixel-level.

    Parameters
    ----------
    arr : np.ndarray, shape (6, H, W), float32
        Downscaled 6-band image, may contain NaN values.

    Returns
    -------
    arr_filled : np.ndarray, shape (6, H, W), float32
        Input array with NaN replaced by 0.0.
    quality    : np.ndarray, shape (6, H, W), float32
        Binary quality map. 1.0 = valid, 0.0 = NaN.
    """
    quality = np.isfinite(arr).astype(np.float32)
    arr_filled = np.where(np.isfinite(arr), arr, 0.0)
    return arr_filled, quality


def preprocess_fits(band_fits_paths, downscale_factor=8):
    """
    Full preprocessing pipeline for a single SPHEREx observation from FITS files.

    Reads up to 6 FITS files (one per band), processes each through
    _get_image_from_fits(), then applies _array_to_quality(), robust_scale(),
    and build_4ch() to produce the model input tensor.

    Missing bands (KeyError or empty filepath) are silently filled with NaN,
    which propagates to quality=0 for those regions.

    Parameters
    ----------
    band_fits_paths : dict {int: str or Path}
        Mapping from band ID (1-6) to FITS file path.
        Missing bands can be omitted or mapped to an empty string "".
    downscale_factor: int (default 8)
        Spatial downscaling factor. 8 -> 2040 becomes 255.

    Returns
    -------
    img_4ch : np.ndarray, shape (4, 255, 765), float32
        Model-ready 4-channel input tensor.
    """
    ds_size = int(np.ceil(2040 / downscale_factor))
    arr = np.full((6, ds_size, ds_size), np.nan, dtype=np.float32)

    for band_id in range(1, 7):
        try:
            fp = band_fits_paths[band_id]
            if not fp:                                 # skip empty filepath
                continue
            arr[band_id - 1] = _get_image_from_fits(fp, downscale_factor)
        except KeyError:
            pass                                       # missing band -> NaN

    down_img, quality = _array_to_quality(arr)
    scaled = robust_scale(down_img)
    return build_4ch(scaled, quality)


def parse_fits_dir(fits_dir):
    """
    Scan a directory for SPHEREx Level 2 FITS files and group them by
    observation ID and band ID.

    Expected filename format:
        level2_{obs_id}D{band_id}_spx_l2b-{version}.fits
    Example:
        level2_2025W39_1A_0001_1D3_spx_l2b-v20-2025-273.fits
          -> obs_id = "2025W39_1A_0001_1", band_id = 3

    Parameters
    ----------
    fits_dir : str or Path
        Directory containing level2_*.fits files.

    Returns
    -------
    obs_ids     : list of str
        Sorted list of unique observation IDs.
    fits_by_obs : dict {obs_id (str): {band_id (int): Path}}
        Nested mapping from obs_id and band_id to FITS file path.
    """
    fits_paths = sorted(Path(fits_dir).glob("level2_*.fits"))
    fits_by_obs = defaultdict(dict)

    for fp in fits_paths:
        parts = fp.name.split("_")
        exposure_id = "_".join(parts[1:5])             # e.g. 2025W39_1A_0001_1
        obs_id, band_id = exposure_id.split("D")       # split on band delimiter
        fits_by_obs[obs_id][int(band_id)] = fp

    obs_ids = sorted(fits_by_obs.keys())
    return obs_ids, dict(fits_by_obs)


# ==============================================================================
# Model
# ==============================================================================

class AttentionBlock(nn.Module):
    """
    Transformer encoder block with multi-head self-attention and MLP.

    Implements the standard ViT attention block:
      x = x + MultiHeadSelfAttention(LayerNorm(x))
      x = x + MLP(LayerNorm(x))

    When return_attn=True, the attention weight matrix is returned with
    retain_grad() enabled, allowing Grad-Attention map computation via
    backpropagation.

    Parameters
    ----------
    embed_dim  : int  — token embedding dimension
    hidden_dim : int  — MLP hidden dimension (typically 2x embed_dim)
    num_heads  : int  — number of attention heads
    dropout    : float — dropout probability (default 0.1)
    """

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5             # 1/sqrt(head_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        B, T, E = x.shape
        h = self.norm1(x)

        # Compute Q, K, V from a single linear projection
        qkv = self.qkv(h).reshape(
            B, T, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_out = attn @ v
        attn_out = attn_out.transpose(1, 2).reshape(B, T, E)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        if return_attn:
            # retain_grad() allows gradient computation on non-leaf tensors
            # required for Grad-Attention map computation
            attn.retain_grad()
            return x, attn

        return x


class VisionTransformerBinaryClassifier(nn.Module):
    """
    Vision Transformer (ViT) for binary classification of SPHEREx images.

    Architecture:
      1. Patchify: split (B, 4, 255, 765) into (B, 867, 900) patch tokens
         where 867 = 17x51 patches of size 15x15, each with 4*15*15=900 dims.
      2. Linear patch embedding -> embed_dim tokens.
      3. Prepend CLS token; add learned positional embeddings.
      4. Pass through num_layers AttentionBlocks.
      5. Extract CLS token -> LayerNorm -> Linear(1) -> scalar logit.

    The last AttentionBlock can optionally return its attention weights
    for Grad-Attention interpretability analysis.

    Parameters
    ----------
    embed_dim    : int   — token embedding dimension (default 128)
    hidden_dim   : int   — MLP hidden dimension (default 256)
    num_channels : int   — input image channels (default 4)
    num_heads    : int   — attention heads per block (default 4)
    num_layers   : int   — number of transformer blocks (default 4)
    patch_size   : int   — spatial patch size in pixels (default 15)
    img_height   : int   — input image height (default 255)
    img_width    : int   — input image width (default 765 = 255*3)
    dropout      : float — dropout probability (default 0.1)
    """

    def __init__(
        self,
        embed_dim=128,
        hidden_dim=256,
        num_channels=4,
        num_heads=4,
        num_layers=4,
        patch_size=15,
        img_height=255,
        img_width=765,
        dropout=0.1
    ):
        super().__init__()

        assert img_height % patch_size == 0
        assert img_width % patch_size == 0

        self.patch_size = patch_size
        self.grid_h = img_height // patch_size         # 17
        self.grid_w = img_width // patch_size          # 51
        self.num_patches = self.grid_h * self.grid_w   # 867

        # Linear patch embedding: (patch_dim) -> embed_dim
        self.input_layer = nn.Linear(
            num_channels * patch_size * patch_size,    # 4 * 15 * 15 = 900
            embed_dim
        )

        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Classification head: LayerNorm -> Linear(1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Truncated normal init for CLS token and positional embeddings;
        Xavier uniform for patch embedding weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

    def _patchify(self, x):
        """
        Split image tensor into flattened patch tokens.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)

        Returns
        -------
        patches : torch.Tensor, shape (B, num_patches, C*P*P)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)               # (B, gh, gw, C, P, P)
        x = x.reshape(B, -1, C * P * P)               # (B, num_patches, patch_dim)
        return x

    def forward(self, x, return_attn=False):
        """
        Forward pass.

        Parameters
        ----------
        x           : torch.Tensor, shape (B, 4, 255, 765)
        return_attn : bool — if True, also return last layer attention weights

        Returns
        -------
        logits      : torch.Tensor, shape (B,) — raw (pre-sigmoid) scores
        last_attn   : torch.Tensor, shape (B, heads, T, T) — only if return_attn=True
        """
        x = self._patchify(x)
        x = self.input_layer(x)

        B, T, E = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                # prepend CLS token
        x = x + self.pos_embedding[:, :T+1]           # add positional embeddings
        x = self.dropout(x)

        last_attn = None
        for i, block in enumerate(self.blocks):
            # Only the last block returns attention (for Grad-Attention)
            if return_attn and i == len(self.blocks) - 1:
                x, last_attn = block(x, return_attn=True)
            else:
                x = block(x)

        cls_out = x[:, 0]                             # CLS token output
        logits = self.head(cls_out).squeeze(-1)        # (B,)

        if return_attn:
            return logits, last_attn

        return logits


def load_model(ckpt_path):
    """
    Load a pretrained VisionTransformerBinaryClassifier from a PyTorch
    Lightning checkpoint file (.ckpt) and return it in eval mode.

    The checkpoint is expected to have been saved by LitViTBinaryClassifier,
    where model weights are stored with a "model." prefix in state_dict keys.
    This function strips the prefix automatically.

    Parameters
    ----------
    ckpt_path : str
        Path to .ckpt file saved by PyTorch Lightning.

    Returns
    -------
    model : VisionTransformerBinaryClassifier
        Pretrained model in eval mode, on CPU.
    """
    model = VisionTransformerBinaryClassifier(
        img_height=255,
        img_width=765,
        patch_size=15,
        num_channels=4,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Strip "model." prefix added by PyTorch Lightning wrapper
    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Model loaded: {ckpt_path}")
    print(f"   Parameters : {sum(p.numel() for p in model.parameters()):,}")

    return model


# ==============================================================================
# Dataset & DataLoader
# ==============================================================================

class SpherexInferenceDataset(Dataset):
    """
    PyTorch Dataset for SPHEREx .npy image cubes.

    Each item is preprocessed on-the-fly via preprocess_file().
    Returns (image_tensor, file_path) per sample.

    Parameters
    ----------
    file_list : list of str
        List of paths to .npy files of shape (6, 2040, 2040).
    """

    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img = preprocess_file(file_path)               # (4, 255, 765)
        return torch.from_numpy(img), file_path


class SpherexFitsDataset(Dataset):
    """
    PyTorch Dataset for SPHEREx Level 2 FITS files.

    Scans fits_dir for level2_*.fits files, groups by observation ID,
    and preprocesses each 6-band observation on-the-fly via preprocess_fits().

    Returns (image_tensor, obs_id, paths) per sample, where paths is a list
    of 6 filepath strings (one per band, empty string if missing).

    Parameters
    ----------
    fits_dir        : str
        Directory containing level2_*.fits files.
    downscale_factor: int (default 8)
        Spatial downscaling factor applied during preprocessing.
    """

    def __init__(self, fits_dir, downscale_factor=8):
        self.downscale_factor = downscale_factor
        self.obs_ids, self.fits_by_obs = parse_fits_dir(fits_dir)

    def __len__(self):
        return len(self.obs_ids)

    def __getitem__(self, idx):
        obs_id = self.obs_ids[idx]
        band_fits_paths = self.fits_by_obs[obs_id]
        img = preprocess_fits(band_fits_paths, self.downscale_factor)
        # Return 6 filepaths as strings (empty string if band missing)
        paths = [str(band_fits_paths.get(b, "")) for b in range(1, 7)]
        return torch.from_numpy(img), obs_id, paths


def build_dataloader(file_list, batch_size=8, num_workers=8):
    """
    Build a DataLoader from a list of .npy file paths.

    Preprocessing is performed on-the-fly by worker processes.
    For CPU inference, num_workers=8 on an 8-core machine is recommended.

    Parameters
    ----------
    file_list   : list of str
    batch_size  : int (default 8)
    num_workers : int (default 8)

    Returns
    -------
    DataLoader
    """
    ds = SpherexInferenceDataset(file_list)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_fits_dataloader(fits_dir, batch_size=8, num_workers=8, downscale_factor=8):
    """
    Build a DataLoader from a directory of SPHEREx Level 2 FITS files.

    Scans fits_dir, groups files by observation ID, and creates a DataLoader
    where each sample is one observation (6 bands preprocessed on-the-fly).

    Parameters
    ----------
    fits_dir        : str
    batch_size      : int (default 8)
    num_workers     : int (default 8)
    downscale_factor: int (default 8) — 2040 -> 255

    Returns
    -------
    DataLoader
    """
    ds = SpherexFitsDataset(fits_dir, downscale_factor=downscale_factor)
    print(f"✅ Found {len(ds)} observations in {fits_dir}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(model, loader, device="cpu"):
    """
    Run silent inference on all batches in a DataLoader (.npy input).

    Parameters
    ----------
    model  : VisionTransformerBinaryClassifier
    loader : DataLoader (SpherexInferenceDataset)
    device : str (default "cpu", use "cuda" for GPU)

    Returns
    -------
    fnames : list of str — file paths, one per sample
    probs  : np.ndarray, shape (N,) — anomaly probabilities in [0, 1]
    """
    model.eval()
    model = model.to(device)

    fnames_all = []
    probs_all = []

    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu()
            fnames_all.extend(fnames)
            probs_all.append(probs.numpy())

    return fnames_all, np.concatenate(probs_all)


def run_inference_tqdm(model, loader, device="cpu"):
    """
    Run inference with a file-level tqdm progress bar (.npy input).

    Parameters
    ----------
    model  : VisionTransformerBinaryClassifier
    loader : DataLoader (SpherexInferenceDataset)
    device : str (default "cpu", use "cuda" for GPU)

    Returns
    -------
    fnames : list of str — file paths, one per sample
    probs  : np.ndarray, shape (N,) — anomaly probabilities in [0, 1]
    """
    from tqdm import tqdm

    model.eval()
    model = model.to(device)

    fnames_all = []
    probs_all = []
    n_files = len(loader.dataset)

    with torch.no_grad():
        with tqdm(total=n_files, desc="Inference", unit="files") as pbar:
            for imgs, fnames in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                fnames_all.extend(fnames)
                probs_all.append(probs.numpy())
                pbar.update(len(fnames))

    return fnames_all, np.concatenate(probs_all)


def run_inference_fits_tqdm(model, loader, device="cpu"):
    """
    Run inference with a file-level tqdm progress bar (FITS input).

    Unlike run_inference_tqdm(), this function also collects the 6 FITS
    filepaths per observation, returned as a list of dicts for easy
    DataFrame construction.

    Parameters
    ----------
    model  : VisionTransformerBinaryClassifier
    loader : DataLoader (SpherexFitsDataset)
    device : str (default "cpu", use "cuda" for GPU)

    Returns
    -------
    obs_ids    : list of str — observation IDs, one per sample
    probs      : np.ndarray, shape (N,) — anomaly probabilities in [0, 1]
    paths_list : list of dict {band_id (int): filepath (str)}
                 One dict per sample, mapping band 1-6 to FITS file paths.
    """
    from tqdm import tqdm

    model.eval()
    model = model.to(device)

    obs_ids_all = []
    probs_all   = []
    paths_all   = []
    n_files = len(loader.dataset)

    with torch.no_grad():
        with tqdm(total=n_files, desc="Inference", unit="files") as pbar:
            for imgs, obs_ids, paths in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                obs_ids_all.extend(obs_ids)
                probs_all.append(probs.numpy())
                # DataLoader collates paths as list of 6 tuples (one per band)
                batch_size = len(obs_ids)
                for i in range(batch_size):
                    paths_all.append({b+1: paths[b][i] for b in range(6)})
                pbar.update(batch_size)

    return obs_ids_all, np.concatenate(probs_all), paths_all


# ==============================================================================
# Grad-Attention — .npy input
# ==============================================================================

def compute_grad_attention_maps(model, file_path, device="cpu", patch_size=15):
    """
    Compute Grad-Attention maps for a single .npy observation.

    Grad-Attention = Gradient x Attention Weight (CLS -> patch, last layer).
    Positive values indicate regions that push the prediction toward anomaly;
    negative values indicate regions that push toward normal.

    Method:
      1. Forward pass with return_attn=True to get attention weights A.
      2. Backpropagate logit.sum() to get dL/dA.
      3. Grad-Attention = A * (dL/dA), averaged over heads.
      4. Separate into positive (pos_map) and negative (neg_map) components.
      5. Reshape from patch grid to image size via bilinear interpolation.

    Parameters
    ----------
    model      : VisionTransformerBinaryClassifier
    file_path  : str — path to .npy file
    device     : str (default "cpu")
    patch_size : int (default 15)

    Returns
    -------
    prob    : float — anomaly probability in [0, 1]
    pos_map : np.ndarray, shape (H, W) — anomaly evidence (positive gradient)
    neg_map : np.ndarray, shape (H, W) — normal evidence  (negative gradient)
    """
    model.eval()
    model = model.to(device)

    img = preprocess_file(file_path)
    xb = torch.from_numpy(img).unsqueeze(0).to(device) # (1, 4, 255, 765)

    B, C, H, W = xb.shape
    grid_h = H // patch_size                           # 17
    grid_w = W // patch_size                           # 51

    model.zero_grad(set_to_none=True)

    logits, attn = model(xb, return_attn=True)         # attn: (1, heads, T, T)
    logits.sum().backward()

    attn_grad = attn.grad                              # (1, heads, T, T)

    # CLS token -> patch attention: slice off CLS row, skip CLS column
    attn_cls = attn[:, :, 0, 1:]                       # (1, heads, num_patches)
    grad_cls  = attn_grad[:, :, 0, 1:]                # (1, heads, num_patches)

    # Element-wise product, average over heads
    grad_attn = (attn_cls * grad_cls).mean(dim=1)      # (1, num_patches)

    # Separate positive (anomaly) and negative (normal) evidence
    pos = torch.clamp( grad_attn, min=0.0).reshape(1, grid_h, grid_w)
    neg = torch.clamp(-grad_attn, min=0.0).reshape(1, grid_h, grid_w)

    # Upsample patch-grid maps to full image resolution
    pos = F.interpolate(pos.unsqueeze(1), size=(H, W),
                        mode="bilinear", align_corners=False).squeeze()
    neg = F.interpolate(neg.unsqueeze(1), size=(H, W),
                        mode="bilinear", align_corners=False).squeeze()

    prob = torch.sigmoid(logits).item()

    return prob, pos.detach().cpu().numpy(), neg.detach().cpu().numpy()


def visualize_grad_attention(
    file_path,
    pos_map,
    neg_map,
    prob=None,
    zclip=(-5, 5),
    overlay_alpha=0.4,
    nan_color="darkorchid",
):
    """
    Visualize 4-channel model input overlaid with Grad-Attention maps (.npy).

    Displays the preprocessed 4-channel image (bandA, bandB, qualA, qualB)
    with two semi-transparent overlays:
      - Red  : positive Grad-Attention (anomaly evidence)
      - Blue : negative Grad-Attention (normal evidence)

    Parameters
    ----------
    file_path     : str — path to .npy file
    pos_map       : np.ndarray, shape (H, W) — from compute_grad_attention_maps()
    neg_map       : np.ndarray, shape (H, W) — from compute_grad_attention_maps()
    prob          : float (optional) — anomaly probability for title
    zclip         : tuple (vmin, vmax) for band channels (default (-5, 5))
    overlay_alpha : float — transparency of attention overlays (default 0.4)
    nan_color     : str — color for masked NaN pixels (default "darkorchid")
    """
    img_4ch = preprocess_file(file_path)
    obsid = os.path.splitext(os.path.basename(file_path))[0]

    zmin, zmax = zclip
    names = ["bandA", "bandB", "qualA", "qualB"]
    title = obsid
    if prob is not None:
        title += f"  |  anomaly prob = {prob:.4f}"

    def _norm(m):
        """Normalize map to [0, 1] for overlay display."""
        m = m - m.min()
        return m / (m.max() + 1e-8)

    pos_norm = _norm(pos_map)
    neg_norm = _norm(neg_map)

    cmap_gray = plt.get_cmap("gray").copy()
    cmap_gray.set_bad(color=nan_color)

    fig, axes = plt.subplots(4, 1, figsize=(12, 4 * 4))
    fig.suptitle(title, fontsize=12)

    for ch in range(4):
        ax = axes[ch]
        img = img_4ch[ch]
        vmin, vmax = (zmin, zmax) if ch < 2 else (0.0, 1.0)
        img_show = np.ma.masked_invalid(img) if ch < 2 else img

        ax.imshow(img_show, origin="lower", cmap=cmap_gray,
                  vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.imshow(pos_norm, origin="lower", cmap="Reds",
                  alpha=overlay_alpha, aspect="equal")
        ax.imshow(neg_norm, origin="lower", cmap="Blues",
                  alpha=overlay_alpha, aspect="equal")
        ax.set_title(names[ch], fontsize=10)
        ax.axis("off")

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="red",  alpha=0.6, label="Anomaly evidence (+)"),
        Patch(facecolor="blue", alpha=0.6, label="Normal evidence (−)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


# ==============================================================================
# Visualization — .npy input
# ==============================================================================

def visualize_raw_bands(
    file_path,
    prob=None,
    zclip=(-3, 3),
    hist_bins=100,
    nan_color="darkorchid",
    text_color="lime",
):
    """
    Visualize raw .npy image: 6 bands split into 2 groups of 3.

    For each band, displays:
      - Robust z-score image (Median MAD scaled, clipped to zclip)
      - Pixel intensity histogram (log-scale) with zclip markers
      - NaN fraction and clip fraction annotated on image

    Layout: 4 rows x 3 cols (image / histogram / image / histogram)
    Bands 1-3 in top group, bands 4-6 in bottom group.

    Parameters
    ----------
    file_path : str — path to .npy file of shape (6, 2040, 2040)
    prob      : float (optional) — anomaly probability for title
    zclip     : tuple (vmin, vmax) for z-score display (default (-3, 3))
    hist_bins : int — number of histogram bins (default 100)
    nan_color : str — color for NaN-masked pixels (default "darkorchid")
    text_color: str — color for overlay text (default "lime")
    """
    arr = np.load(file_path).astype(float)
    obsid = os.path.splitext(os.path.basename(file_path))[0]
    n_channels = arr.shape[0]

    zmin, zmax = zclip
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(color=nan_color)

    fig, axes = plt.subplots(
        4, 3,
        figsize=(3.0 * 3, 1.95 * 4),
        squeeze=False,
        gridspec_kw={"height_ratios": [2, 0.35, 2, 0.35]}
    )
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05,
                        wspace=0.02, hspace=0.01)

    title = obsid
    if prob is not None:
        title += f"  |  anomaly prob = {prob:.4f}"
    fig.suptitle(title, fontsize=12)

    for ch in range(n_channels):
        col      = ch % 3
        row_img  = (ch // 3) * 2
        row_hist = (ch // 3) * 2 + 1

        band = arr[ch]
        finite = band[np.isfinite(band)]
        nan_frac = np.isnan(band).mean()

        ax_img  = axes[row_img,  col]
        ax_hist = axes[row_hist, col]

        # All-NaN band: display black panel with label
        if finite.size == 0:
            ax_img.set_facecolor("black")
            ax_img.text(0.5, 0.5, "ALL NaN", transform=ax_img.transAxes,
                        ha="center", va="center", fontsize=12,
                        color="darkorchid", fontweight="bold")
            ax_img.axis("off")
            ax_hist.axis("off")
            continue

        # Robust z-score scaling (Median MAD)
        m = np.median(finite)
        mad = np.median(np.abs(finite - m))
        sigma = 1.4826 * mad + 1e-6
        z = (band - m) / sigma
        clipped_frac = np.mean(np.abs(z) > zmax)
        z_ma = np.ma.masked_invalid(np.clip(z, zmin, zmax))

        ax_img.imshow(z_ma, origin="lower", cmap=cmap,
                      vmin=zmin, vmax=zmax, interpolation="nearest")
        ax_img.axis("off")

        for txt, y in [(f"nan={nan_frac:.2f}", 0.98), (f"clip={clipped_frac:.2f}", 0.88)]:
            ax_img.text(0.02, y, txt, transform=ax_img.transAxes,
                        ha="left", va="top", fontsize=9,
                        color=text_color, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        # Histogram
        z_finite = z[np.isfinite(z)]
        ax_hist.hist(z_finite, bins=hist_bins, color="gray")
        ax_hist.set_yscale("log")
        ax_hist.axvline(zmin, color="red",  linestyle="--", linewidth=1)
        ax_hist.axvline(zmax, color="blue", linestyle="--", linewidth=1)
        ax_hist.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_hist.tick_params(axis="x", labelsize=7)

    plt.show()


def visualize_model_input(
    file_path,
    prob=None,
    zclip=(-5, 5),
    nan_color="darkorchid",
    cmap="gray",
):
    """
    Visualize the preprocessed 4-channel model input from a .npy file.

    Displays bandA, bandB (z-score scaled), qualA, qualB (quality [0,1])
    as 4 vertically stacked panels, each with 3:1 aspect ratio.

    Parameters
    ----------
    file_path : str — path to .npy file
    prob      : float (optional) — anomaly probability for title
    zclip     : tuple (vmin, vmax) for band channels (default (-5, 5))
    nan_color : str — color for NaN-masked pixels (default "darkorchid")
    cmap      : str — colormap name (default "gray")
    """
    img_4ch = preprocess_file(file_path)
    obsid = os.path.splitext(os.path.basename(file_path))[0]

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=nan_color)

    names = ["bandA", "bandB", "qualA", "qualB"]
    zmin, zmax = zclip
    title = obsid
    if prob is not None:
        title += f"  |  anomaly prob = {prob:.4f}"

    fig, axes = plt.subplots(4, 1, figsize=(12, 4 * 4))
    fig.suptitle(title, fontsize=12)

    for ch in range(4):
        ax = axes[ch]
        img = img_4ch[ch]
        vmin, vmax = (zmin, zmax) if ch < 2 else (0.0, 1.0)
        img_show = np.ma.masked_invalid(img) if ch < 2 else img

        ax.imshow(img_show, origin="lower", cmap=cmap_obj,
                  vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.set_title(names[ch], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Visualization — FITS input
# ==============================================================================

def _parse_obs_id_from_paths(band_paths):
    """Extract obs_id string from the first available FITS filepath."""
    for b in range(1, 7):
        fp = band_paths.get(b, "")
        if fp:
            parts = Path(fp).name.split("_")
            return "_".join(parts[1:5])                # e.g. 2025W39_1A_0001_1
    return "unknown"


def visualize_raw_bands_fits(
    band_paths,
    prob=None,
    zclip=(-3, 3),
    hist_bins=100,
    nan_color="darkorchid",
    text_color="lime",
):
    """
    Visualize raw SPHEREx FITS images: 6 bands split into 2 groups of 3.

    Reads each band directly from FITS (IMAGE HDU, after flagging and
    ZODI subtraction), then displays z-score images and histograms.
    Missing or empty-path bands are shown as black "MISSING" panels.

    Parameters
    ----------
    band_paths : dict {int: str} — band ID (1-6) to FITS filepath
    prob       : float (optional) — anomaly probability for title
    zclip      : tuple (vmin, vmax) for z-score display (default (-3, 3))
    hist_bins  : int — number of histogram bins (default 100)
    nan_color  : str — color for NaN-masked pixels (default "darkorchid")
    text_color : str — color for overlay text (default "lime")
    """
    from astropy.io import fits as afits

    zmin, zmax = zclip
    obs_id = _parse_obs_id_from_paths(band_paths)
    title = obs_id
    if prob is not None:
        title += f"  |  anomaly prob = {prob:.4f}"

    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(color=nan_color)

    fig, axes = plt.subplots(
        4, 3,
        figsize=(3.0 * 3, 1.95 * 4),
        squeeze=False,
        gridspec_kw={"height_ratios": [2, 0.35, 2, 0.35]}
    )
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05,
                        wspace=0.02, hspace=0.01)
    fig.suptitle(title, fontsize=12)

    for band_id in range(1, 7):
        ch = band_id - 1
        col      = ch % 3
        row_img  = (ch // 3) * 2
        row_hist = (ch // 3) * 2 + 1

        ax_img  = axes[row_img,  col]
        ax_hist = axes[row_hist, col]

        fp = band_paths.get(band_id, "")

        # Missing band
        if not fp:
            ax_img.set_facecolor("black")
            ax_img.text(0.5, 0.5, "MISSING", transform=ax_img.transAxes,
                        ha="center", va="center", fontsize=12,
                        color="darkorchid", fontweight="bold")
            ax_img.axis("off")
            ax_hist.axis("off")
            continue

        # Read FITS
        with afits.open(fp) as hdulist:
            image = hdulist["IMAGE"].data.astype(float)
            flags = hdulist["FLAGS"].data
            zodi  = hdulist["ZODI"].data
            flag_val = (
                7
                + (1 << hdulist["FLAGS"].header["MP_SOURCE"])
                + (1 << hdulist["FLAGS"].header["MP_NONFUNC"])
            )
            image[(flags & flag_val) > 0] = np.nan
            image -= zodi

        band = image
        finite = band[np.isfinite(band)]
        nan_frac = np.isnan(band).mean()

        # All-NaN band
        if finite.size == 0:
            ax_img.set_facecolor("black")
            ax_img.text(0.5, 0.5, "ALL NaN", transform=ax_img.transAxes,
                        ha="center", va="center", fontsize=12,
                        color="darkorchid", fontweight="bold")
            ax_img.axis("off")
            ax_hist.axis("off")
            continue

        # Robust z-score
        m = np.median(finite)
        mad = np.median(np.abs(finite - m))
        sigma = 1.4826 * mad + 1e-6
        z = (band - m) / sigma
        clipped_frac = np.mean(np.abs(z) > zmax)
        z_ma = np.ma.masked_invalid(np.clip(z, zmin, zmax))

        ax_img.imshow(z_ma, origin="lower", cmap=cmap,
                      vmin=zmin, vmax=zmax, interpolation="nearest")
        ax_img.axis("off")

        for txt, y in [(f"nan={nan_frac:.2f}", 0.98), (f"clip={clipped_frac:.2f}", 0.88)]:
            ax_img.text(0.02, y, txt, transform=ax_img.transAxes,
                        ha="left", va="top", fontsize=9,
                        color=text_color, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        z_finite = z[np.isfinite(z)]
        ax_hist.hist(z_finite, bins=hist_bins, color="gray")
        ax_hist.set_yscale("log")
        ax_hist.axvline(zmin, color="red",  linestyle="--", linewidth=1)
        ax_hist.axvline(zmax, color="blue", linestyle="--", linewidth=1)
        ax_hist.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_hist.tick_params(axis="x", labelsize=7)

    plt.show()


def visualize_model_input_fits(
    band_paths,
    prob=None,
    zclip=(-5, 5),
    nan_color="darkorchid",
    cmap="gray",
):
    """
    Visualize the preprocessed 4-channel model input from FITS files.

    Parameters
    ----------
    band_paths : dict {int: str} — band ID (1-6) to FITS filepath
    prob       : float (optional) — anomaly probability for title
    zclip      : tuple (vmin, vmax) for band channels (default (-5, 5))
    nan_color  : str — color for NaN-masked pixels (default "darkorchid")
    cmap       : str — colormap name (default "gray")
    """
    img_4ch = preprocess_fits(band_paths)
    obs_id = _parse_obs_id_from_paths(band_paths)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=nan_color)

    names = ["bandA", "bandB", "qualA", "qualB"]
    zmin, zmax = zclip
    title = obs_id
    if prob is not None:
        title += f"  |  anomaly prob = {prob:.4f}"

    fig, axes = plt.subplots(4, 1, figsize=(12, 4 * 4))
    fig.suptitle(title, fontsize=12)

    for ch in range(4):
        ax = axes[ch]
        img = img_4ch[ch]
        vmin, vmax = (zmin, zmax) if ch < 2 else (0.0, 1.0)
        img_show = np.ma.masked_invalid(img) if ch < 2 else img

        ax.imshow(img_show, origin="lower", cmap=cmap_obj,
                  vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.set_title(names[ch], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_grad_attention_fits(
    model,
    band_paths,
    prob=None,
    device="cpu",
    patch_size=15,
    zclip=(-5, 5),
    overlay_alpha=0.4,
    nan_color="darkorchid",
):
    """
    Compute and visualize Grad-Attention maps from FITS files.

    Internally calls preprocess_fits() to build the model input, then
    computes Grad-Attention via backpropagation on the logit sum.
    Displays the 4-channel input overlaid with:
      - Red  : positive Grad-Attention (anomaly evidence)
      - Blue : negative Grad-Attention (normal evidence)

    Parameters
    ----------
    model      : VisionTransformerBinaryClassifier
    band_paths : dict {int: str} — band ID (1-6) to FITS filepath
    prob       : float (optional) — if None, computed from model forward pass
    device     : str (default "cpu")
    patch_size : int (default 15)
    zclip      : tuple (vmin, vmax) for band channels (default (-5, 5))
    overlay_alpha : float — transparency of attention overlays (default 0.4)
    nan_color  : str — color for NaN-masked pixels (default "darkorchid")
    """
    img = preprocess_fits(band_paths)
    xb = torch.from_numpy(img).unsqueeze(0).to(device)

    B, C, H, W = xb.shape
    grid_h = H // patch_size
    grid_w = W // patch_size

    model.eval()
    model = model.to(device)
    model.zero_grad(set_to_none=True)

    logits, attn = model(xb, return_attn=True)
    logits.sum().backward()

    attn_grad = attn.grad
    attn_cls  = attn[:, :, 0, 1:]
    grad_cls  = attn_grad[:, :, 0, 1:]
    grad_attn = (attn_cls * grad_cls).mean(dim=1)

    pos = torch.clamp( grad_attn, min=0.0).reshape(1, grid_h, grid_w)
    neg = torch.clamp(-grad_attn, min=0.0).reshape(1, grid_h, grid_w)

    pos = F.interpolate(pos.unsqueeze(1), size=(H, W),
                        mode="bilinear", align_corners=False).squeeze()
    neg = F.interpolate(neg.unsqueeze(1), size=(H, W),
                        mode="bilinear", align_corners=False).squeeze()

    if prob is None:
        prob = torch.sigmoid(logits).item()

    pos_map = pos.detach().cpu().numpy()
    neg_map = neg.detach().cpu().numpy()

    obs_id = _parse_obs_id_from_paths(band_paths)
    zmin, zmax = zclip
    names = ["bandA", "bandB", "qualA", "qualB"]
    title = f"{obs_id}  |  anomaly prob = {prob:.4f}"

    def _norm(m):
        m = m - m.min()
        return m / (m.max() + 1e-8)

    pos_norm = _norm(pos_map)
    neg_norm = _norm(neg_map)

    cmap_gray = plt.get_cmap("gray").copy()
    cmap_gray.set_bad(color=nan_color)

    fig, axes = plt.subplots(4, 1, figsize=(12, 4 * 4))
    fig.suptitle(title, fontsize=12)

    for ch in range(4):
        ax = axes[ch]
        img_ch = img[ch]
        vmin, vmax = (zmin, zmax) if ch < 2 else (0.0, 1.0)
        img_show = np.ma.masked_invalid(img_ch) if ch < 2 else img_ch

        ax.imshow(img_show, origin="lower", cmap=cmap_gray,
                  vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.imshow(pos_norm, origin="lower", cmap="Reds",
                  alpha=overlay_alpha, aspect="equal")
        ax.imshow(neg_norm, origin="lower", cmap="Blues",
                  alpha=overlay_alpha, aspect="equal")
        ax.set_title(names[ch], fontsize=10)
        ax.axis("off")

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="red",  alpha=0.6, label="Anomaly evidence (+)"),
        Patch(facecolor="blue", alpha=0.6, label="Normal evidence (−)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
