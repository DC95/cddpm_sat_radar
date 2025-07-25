# deepconvectivecore_ddpm_train.py
# DDPM training pipeline for conditional IR → DPR reflectivity mapping

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import yaml
import argparse
import datetime
import shutil
from torchinfo import summary
import time
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont
from torchvision.utils import make_grid

from diffusion_utils import (
    cosine_beta_schedule,
    extract,
    q_sample,
    p_losses,
    safe_p_losses,
    set_seed,
    XArrayCGANDataset,
    setup_logging,
    save_fixed_batch,
    load_fixed_batch,
    parameter_summary,
    load_config,
    load_latest_checkpoint,
    save_ddpm_sample
)

from conditional_unet_meanonly import ConditionalDenoisingUNet


def save_config_copy(original_config_path, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, "config_copy.yaml")
    try:
        shutil.copy2(original_config_path, dest_path)
        logger.info(f"[INFO] Saved config copy to: {dest_path}")
    except Exception as e:
        logger.warning(f"[WARN] Failed to copy config: {e}")


def main(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 42))
    # -------------------------------------------------
    # TensorBoard for tracking training
    # -------------------------------------------------
    writer = SummaryWriter(log_dir=os.path.join(args.output_path, "tensorboard_logs")) if args.tensorboard else None

    # -------------------- Dataset ----------------------
    dataset = XArrayCGANDataset(
        xr_file_path=config["data"]["input_path"],
        condition_vars=config["variables"]["condition_vars"],
        selected_condition_vars=config["variables"]["selected_condition_vars"],
        angle_vars=config["variables"]["angle_vars"],
        target_vars=config["variables"]["target_vars"],
        means_condition=config["normalization"]["mean_condition"],
        stds_condition=config["normalization"]["std_condition"],
        means_angle=config["normalization"]["mean_angle"],
        stds_angle=config["normalization"]["std_angle"],
        means_target=config["normalization"]["mean_target"],
        stds_target=config["normalization"]["std_target"],
        target_fill_value=0.0,
        target_threshold=0,
        time_steps=config["variables"]["time_steps"],
        max_patches=None,
        bad_indices=config.get("bad_indices", None)
    )
    dataset.load()
    
    # 80% training, 20% test split
    val_size = 20 * config["training"]["batch_size"]
    train_size = len(dataset) - val_size
    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    # -------------------------------------------------
    # Fixed batch for validation/visual monitoring
    # -------------------------------------------------
    fixed_conditions, fixed_targets = next(iter(test_loader))
    if fixed_conditions.ndim == 5:
        fixed_conditions = fixed_conditions.view(fixed_conditions.shape[0], -1, fixed_conditions.shape[-2], fixed_conditions.shape[-1])
    if fixed_targets.ndim == 5 and fixed_targets.shape[2] == 1:
        fixed_targets = fixed_targets.view(fixed_targets.shape[0], fixed_targets.shape[1], fixed_targets.shape[3], fixed_targets.shape[4])

    fixed_conditions, fixed_targets = fixed_conditions.to(device), fixed_targets.to(device)
    save_fixed_batch(fixed_conditions, fixed_targets, os.path.join(args.output_path, "fixed_batch.pt"))

    # -------------------------------------------------
    # Model initialization
    # -------------------------------------------------
    model = ConditionalDenoisingUNet(
        in_channels=fixed_conditions.shape[1],
        model_channels=config["model"]["channels"],
        out_channels=1 # Only predicting the mean;
    ).to(device)
    parameter_summary(model)
    try:
        summary(model, input_data=(fixed_targets, torch.randint(0, 1000, (fixed_targets.shape[0],)).to(device), fixed_conditions))
    except:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # -------------------------------------------------
    # Optionally resume training
    # -------------------------------------------------
    start_epoch = 0
    if args.resume_path and os.path.exists(args.resume_path):
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"[INFO] Resuming from checkpoint at epoch {start_epoch}")
  
    
    # -------------------------------------------------
    # Diffusion parameters (cosine schedule)
    # -------------------------------------------------
    betas = cosine_beta_schedule(timesteps=config["diffusion"]["timesteps"]).to(device)

    best_val_loss = float("inf")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        running_loss = 0.0

        for conditions, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            conditions, targets = conditions.to(device), targets.to(device)
            if conditions.ndim == 5:
                B, C, T, H, W = conditions.shape
                conditions = conditions.view(B, C*T, H, W)
            if targets.ndim == 5 and targets.shape[2] == 1:
                targets = targets.view(targets.shape[0], targets.shape[1], targets.shape[3], targets.shape[4])

            # Sample random diffusion step
            t = torch.randint(0, config["diffusion"]["timesteps"], (targets.size(0),), device=device).long()
            # Use mean-only loss
            if config["training"].get("loss_safe", False):
                loss = safe_p_losses(model, targets, t, conditions, betas, standardize=True)
            else:    
                loss = p_losses(model, targets, t, conditions, betas, standardize=True)

            if torch.isnan(loss):
                print("[WARN] NaN loss detected; skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            if writer:
                global_step = epoch * len(train_loader)
                writer.add_scalar("Loss/train", loss.item(), global_step)

        avg_loss = running_loss / len(train_loader)
        print(f"[TRAIN] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        scheduler.step()

        # -------------------- Validation ----------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_cond, val_targets in test_loader:
                val_cond, val_targets = val_cond.to(device), val_targets.to(device)
                if val_cond.ndim == 5:
                    val_cond = val_cond.view(val_cond.shape[0], -1, val_cond.shape[-2], val_cond.shape[-1])
                if val_targets.ndim == 5 and val_targets.shape[2] == 1:
                    val_targets = val_targets.view(val_targets.shape[0], val_targets.shape[1], val_targets.shape[3], val_targets.shape[4])
                t = torch.randint(0, config["diffusion"]["timesteps"], (val_targets.size(0),), device=device).long()
                val_loss += p_losses(model, val_targets, t, val_cond, betas, standardize=True).item()

        val_loss /= len(test_loader)
        print(f"[VAL]   Epoch {epoch+1} Avg Loss: {val_loss:.4f}")
        if writer:
            writer.add_scalar("Loss/val", val_loss, epoch)

        # -------------------- Checkpoint ----------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, os.path.join(args.output_path, "best_model.pt"))
            print(f"[INFO] Best model updated at epoch {epoch+1}")

        if epoch % args.save_frequency == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, os.path.join(args.output_path, f"ddpm_checkpoint_epoch_{epoch}.pt"))

            if epoch % 10 == 0:
                logger.info("Starting DDPM sample generation...")
                save_ddpm_sample(model, test_loader, os.path.join(args.output_path, "samples"),
                                epoch, device, config["diffusion"]["timesteps"], betas, config)
                logger.info("Finished DDPM sample generation.")

    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--save_frequency', type=int, default=10)
    parser.add_argument('--resume_path', type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_path, exist_ok=True)
    logger = setup_logging(args.output_path, "train.log")
    save_config_copy(args.config, args.output_path, logger)
    
    main(args, config)

