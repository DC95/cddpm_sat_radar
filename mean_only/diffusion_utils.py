# diffusion_utils.py
# Core utilities for DDPM forward process and training loss

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import csv
import argparse
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import xarray as xr
import yaml
import glob
import torch
import torch.distributed as dist
import sys
import re
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_ddpm_sample(model, test_loader, output_dir, epoch, device, timesteps, betas, config):
    """
    Generate 20 random GT–Prediction pairs using DDPM model on test samples.
    Saves a single image grid with GT–Pred pairs.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    mean = torch.tensor(config["normalization"]["mean_target"], device=device).view(1, -1, 1, 1)
    std = torch.tensor(config["normalization"]["std_target"], device=device).view(1, -1, 1, 1)

    samples_collected = 0
    gt_list, pred_list = [], []

    with torch.no_grad():
        for cond, gt in test_loader:
            cond, gt = cond.to(device), gt.to(device)

            # Collapse time dimension if needed
            if cond.ndim == 5:
                B, C, T, H, W = cond.shape
                cond = cond.view(B, C * T, H, W)
            if gt.ndim == 5 and gt.shape[2] == 1:
                gt = gt.view(gt.shape[0], gt.shape[1], gt.shape[3], gt.shape[4])

            B = gt.size(0)
            remaining = 20 - samples_collected
            take = min(B, remaining)

            # Slice only needed samples
            cond = cond[:take]
            gt = gt[:take]

            x = torch.randn_like(gt)

            # DDPM reverse process
            for t in reversed(range(timesteps)):
                t_tensor = torch.full((take,), t, device=device, dtype=torch.long)
                mean_pred = model(x, t_tensor, cond)
                beta_t = betas[t_tensor].view(take, 1, 1, 1)
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                x = mean_pred + torch.sqrt(beta_t) * noise

            x_denorm = x * std + mean
            gt_denorm = gt * std + mean

            gt_list.append(gt_denorm.cpu())
            pred_list.append(x_denorm.cpu())
            samples_collected += take

            if samples_collected >= 20:
                break

    # Concatenate and build image grid
    gt_all = torch.cat(gt_list, dim=0)[:20]
    pred_all = torch.cat(pred_list, dim=0)[:20]
    interleaved = torch.stack([gt_all, pred_all], dim=1).view(-1, 1, gt_all.shape[2], gt_all.shape[3])

    vmin, vmax = interleaved.min().item(), interleaved.max().item()
    grid = make_grid(interleaved, nrow=2, normalize=True, value_range=(vmin, vmax), pad_value=1)
    img = to_pil_image(grid)

    # Add labels
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i in range(20):
        y_offset = i * (img.height // 20)
        draw.text((5, y_offset + 2), f"Sample {i+1}", fill="white", font=font)

    img.save(os.path.join(output_dir, f"epoch_{epoch:03d}_gt_vs_pred_pairs.png"))
    print(f"[INFO] Saved 20 DDPM GT–Pred pairs at epoch {epoch}")
    
def safe_p_losses(denoise_model, x_start, t, cond, betas, standardize=False):
    """
    Safe loss function with runtime error guards for NaNs and Infs.
    """
    try:
        if standardize:
            mean = x_start.mean(dim=(2, 3), keepdim=True)
            std = x_start.std(dim=(2, 3), keepdim=True) + 1e-5
            x_start = (x_start - mean) / std

        noise = torch.randn_like(x_start)
        alpha = 1. - betas
        alphas_cumprod = torch.cumprod(alpha, dim=0)

        x_noisy = q_sample(x_start, t, noise, alphas_cumprod)
        predicted_noise = denoise_model(x_noisy, t, cond)

        if torch.any(torch.isnan(predicted_noise)) or torch.any(torch.isinf(predicted_noise)):
            raise ValueError("NaN or Inf in predicted noise")

        return F.mse_loss(predicted_noise, noise)
    except Exception as e:
        print(f"[ERROR] Loss computation failed: {e}")
        return torch.tensor(0.0, requires_grad=True, device=x_start.device)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Generates a cosine schedule of beta values, clipped to avoid instability.
    Pros:
        - Smooth decay of noise
        - Avoids abrupt changes compared to linear schedule
    Cons:
        - Needs clipping to avoid extreme beta values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    batch_size = t.size(0)
    out = a.gather(-1, t).float()
    return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    sqrt_alphas_cumprod = torch.sqrt(extract(alphas_cumprod, t, x_start.shape))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - extract(alphas_cumprod, t, x_start.shape))
    return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

def q_sample(x_start, t, noise, alphas_cumprod):
    sqrt_alphas_cumprod = torch.sqrt(extract(alphas_cumprod, t, x_start.shape))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - extract(alphas_cumprod, t, x_start.shape))
    return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

def p_losses(denoise_model, x_start, t, cond, betas, standardize=False):
    """
    Compute the DDPM training loss between predicted noise and true noise.
    Args:
        denoise_model: The U-Net model.
        x_start: The clean input image [B, C, H, W].
        t: Time step tensor [B].
        cond: Conditional input [B, C_cond, H, W].
        betas: Tensor of beta values [T].
        standardize: Whether to standardize x_start before adding noise.
    Returns:
        MSE loss between predicted and actual noise.
    """
    if standardize:
        mean = x_start.mean(dim=(2, 3), keepdim=True)
        std = x_start.std(dim=(2, 3), keepdim=True) + 1e-5
        x_start = (x_start - mean) / std
        print(f"[DEBUG] x_start standardized: mean≈{x_start.mean().item():.2f}, std≈{x_start.std().item():.2f}")

    noise = torch.randn_like(x_start)

    alpha = 1. - betas
    alphas_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    x_noisy = q_sample(x_start, t, noise, alphas_cumprod)
    predicted_noise = denoise_model(x_noisy, t, cond)

    loss = F.mse_loss(predicted_noise, noise)

    if torch.any(torch.isnan(predicted_noise)) or torch.any(torch.isinf(predicted_noise)):
        print("[WARN] NaN or Inf detected in predicted noise")

    print(f"[DEBUG] p_losses: x_start min={x_start.min().item():.2f}, max={x_start.max().item():.2f}, mean={x_start.mean().item():.2f}")
    print(f"[DEBUG] x_noisy min={x_noisy.min().item():.2f}, max={x_noisy.max().item():.2f}, mean={x_noisy.mean().item():.2f}")
    print(f"[DEBUG] predicted_noise min={predicted_noise.min().item():.2f}, max={predicted_noise.max().item():.2f}, mean={predicted_noise.mean().item():.2f}")
    return loss
    
#Data Loader:
class XArrayCGANDataset(Dataset):
    def __init__(self, xr_file_path, condition_vars, selected_condition_vars, angle_vars, target_vars, 
                 means_condition, stds_condition, means_angle, stds_angle, means_target, stds_target, 
                 target_fill_value=0.0, target_threshold=None, time_steps=9, max_patches=None, bad_indices=None,):
        """
        Dataset class for conditional GAN using xarray with lazy loading and modular variable selection.

        :param condition_vars: Full list of all possible condition variables
        :param selected_condition_vars: Subset of variables to load/use
        :param time_steps: Number of time steps to keep (<= 9)
        """
        self.file_path = xr_file_path
        self.condition_vars = condition_vars
        self.selected_condition_vars = selected_condition_vars or condition_vars
        self.angle_vars = angle_vars
        self.target_vars = target_vars
        self.time_steps = time_steps
        self.max_patches = max_patches

        self.transform_condition = transforms.Normalize(mean=means_condition, std=stds_condition)
        self.transform_angle = transforms.Normalize(mean=means_angle, std=stds_angle)
        self.transform_target = NormalizeAndHandleMissing(mean=means_target, std=stds_target,
                                                          fill_value=target_fill_value, threshold=target_threshold)

        self.xr_data = None
        self.bad_indices = set(bad_indices) if bad_indices is not None else None

    def load(self):
        ds = xr.open_dataset(self.file_path, engine='netcdf4')

        # Réduction temporelle centrée
        start_time = (9 - self.time_steps) // 2
        end_time = start_time + self.time_steps
        time_slice = slice(start_time, end_time)

        # Liste complète des indices de patchs disponibles
        total_patches = ds.sizes["patches"]
        all_indices = np.arange(total_patches)

        # Filtrage des indices invalides si spécifiés
        if self.bad_indices is not None:
            valid_indices = [i for i in all_indices if i not in self.bad_indices]
        else:
            valid_indices = list(all_indices)

        # Application de la limite max_patches si définie
        if self.max_patches is not None:
            valid_indices = valid_indices[:self.max_patches]

        # Chargement des données sur les indices filtrés et la fenêtre temporelle choisie
        self.xr_data = {
            "conditions": ds[self.selected_condition_vars].to_array()
                .isel(patches=valid_indices, time=time_slice)
                .transpose("patches", "variable", "x", "y", "time"),

            "angles": ds[self.angle_vars].to_array()
                .isel(patches=valid_indices),

            "targets": ds[self.target_vars].to_array()
                .isel(patches=valid_indices),
        }

        print(f"[INFO] Données chargées : {self.xr_data['conditions'].shape[0]} patches (valide), "
            f"{self.xr_data['conditions'].shape[-1]} steps temporels")
        

    def __len__(self):
        if self.xr_data is None:
            raise RuntimeError("Appelle `.load()` avant d'utiliser le dataset.")
        return self.xr_data["targets"].sizes["patches"]

    def __getitem__(self, idx):
        if self.xr_data is None:
            raise RuntimeError("Appelle `.load()` avant d'utiliser le dataset.")

        condition_np = self.xr_data["conditions"].isel(patches=idx).values  # [C, X, Y, T]
        angle_np = self.xr_data["angles"].isel(patches=idx).values
        target_np = self.xr_data["targets"].isel(patches=idx).values

        condition = torch.tensor(condition_np, dtype=torch.float32).permute(0, 3, 1, 2)  # [C, T, H, W]
        angle = torch.tensor(angle_np, dtype=torch.float32).unsqueeze(1)  # [1, X, Y]
        target = torch.tensor(target_np, dtype=torch.float32).unsqueeze(1)  # [1, X, Y]

        # Normalisation manuelle de condition
        mean_c = torch.tensor(self.transform_condition.mean, dtype=torch.float32)
        std_c = torch.tensor(self.transform_condition.std, dtype=torch.float32)
        condition = (condition - mean_c[:, None, None, None]) / (std_c[:, None, None, None] + 1e-8)

        angle = self.transform_angle(angle)
        target = self.transform_target(target)

        # Intégration de l'angle comme canal temporel
        angle = angle.repeat(1, condition.shape[1], 1, 1)
        condition = torch.cat((condition, angle), dim=0)  # [C+1, T, H, W]

        return condition, target
        
# Configurer le logging
def setup_logging(log_dir, log_filename="train.log", debug=False):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("ddpm_training")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear previous handlers if reinitializing
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG if debug else logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
    
def save_fixed_batch(fixed_conditions, fixed_targets, path):
    """Save fixed_conditions and fixed_targets to disk."""
    torch.save({
        "fixed_conditions": fixed_conditions.cpu(),
        "fixed_targets": fixed_targets.cpu()
    }, path)
    
def load_fixed_batch(path, device):
    """Load fixed_conditions and fixed_targets from disk."""
    data = torch.load(path)
    fixed_conditions = data["fixed_conditions"].to(device)
    fixed_targets = data["fixed_targets"].to(device)
    return fixed_conditions, fixed_targets
    
def parameter_summary(model):
    '''
    1. Iterates over all named parameters (weights, biases) in the model.
    2. Counts parameters for each layer using param.numel() (total elements).
    3. Prints a layer-wise parameter count and total trainable parameters.
    '''
    print(f"{'Layer':<40}{'Parameters':>15}")
    print("=" * 55)
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:<40}{param_count:>15}")
    print("=" * 55)
    print(f"{'Total Parameters':<40}{total_params:>15}")
    
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Get full list and selected list
    all_vars = config["variables"]["condition_vars"]
    selected_vars = config["variables"].get("selected_condition_vars", all_vars)

    # Create index mapping to preserve order
    index_map = {var: idx for idx, var in enumerate(all_vars)}

    # Filter mean_condition and std_condition by selecting corresponding indices
    selected_indices = [index_map[var] for var in selected_vars]

    config["normalization"]["mean_condition"] = [config["normalization"]["mean_condition"][idx] for idx in selected_indices]
    config["normalization"]["std_condition"] = [config["normalization"]["std_condition"][idx] for idx in selected_indices]

    return config
    
def load_latest_checkpoint(output_path, generator, discriminator, optimizer_G, optimizer_D):
    checkpoint_files = glob.glob(os.path.join(output_path, "checkpoint_epoch_*.pth"))

    if not checkpoint_files:
        logging.info("No previous checkpoint found. Starting fresh training.")
        return 0

    # Trier correctement par numéro d’époque
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)
    latest_checkpoint = checkpoint_files[-2]

    checkpoint = torch.load(latest_checkpoint)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

    last_epoch = checkpoint["epoch"] + 1
    logging.info(f"Resuming training from epoch {last_epoch}. Loaded {latest_checkpoint}")
    return last_epoch
    
class NormalizeAndHandleMissing:
    def __init__(self, mean=0.0, std=1.0, fill_value=0.0, threshold=None):
        """
        Parameters:
        - mean (float or torch.Tensor): Mean for normalization.
        - std (float or torch.Tensor): Standard deviation for normalization.
        - fill_value (float): Value to replace NaN and values below the threshold.
        - threshold (float or None): If set, values below this threshold will be replaced with `fill_value`.
        """
        self.mean = torch.tensor(mean, dtype=torch.float32) if not isinstance(mean, torch.Tensor) else mean
        self.std = torch.tensor(std, dtype=torch.float32) if not isinstance(std, torch.Tensor) else std
        self.fill_value = fill_value
        self.threshold = threshold
        self.fill_tensor = torch.tensor(self.fill_value, dtype=torch.float32)
        

    def __call__(self, raw_data):
        """
        Parameters:
        - raw_data (torch.Tensor): The input tensor to normalize and handle.

        Returns:
        - torch.Tensor: Transformed tensor.
        """
        # Replace NaNs with the fill value
        raw_data = torch.nan_to_num(raw_data, nan=self.fill_value)
        
        # Replace values below the threshold with the fill value (if threshold is set)
        if self.threshold is not None:
            raw_data = torch.where(raw_data < self.threshold, self.fill_tensor, raw_data)
        
        # Normalize the data with a safeguard for zero std
        eps = 1e-8
        normalized_data = (raw_data - self.mean) / (self.std + eps)

        return normalized_data
