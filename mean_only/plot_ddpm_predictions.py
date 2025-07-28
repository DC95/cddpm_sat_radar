# python plot_ddpm_predictions.py --config config_ddpm_june.yaml --checkpoint /path/to/checkpoint_epoch190.pt --output_dir /path/to/visualizations/ --device cuda

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from diffusion_utils import load_config, XArrayCGANDataset, cosine_beta_schedule
from conditional_unet_meanonly import ConditionalDenoisingUNet
import argparse

torch.cuda.empty_cache()

def plot_ddpm_sample_pairs(model, dataset, config, output_dir, device, num_samples=5):
    """
    Runs inference with a trained DDPM model on `num_samples` random patches from the dataset.
    Generates a side-by-side visualization of GT vs Predicted reflectivity with:
      - RMSE and SSIM as subplot titles
      - vmin/vmax and color scale consistent across all images

    Saves a single summary figure to disk.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Load normalization constants
    mean = torch.tensor(config["normalization"]["mean_target"], device=device).view(1, -1, 1, 1)
    std = torch.tensor(config["normalization"]["std_target"], device=device).view(1, -1, 1, 1)

    # Random sample indices
    indices = torch.randperm(len(dataset))[:num_samples]

    # Initialize plot
    fig, axs = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))

    # Load beta schedule and timesteps
    T = config["diffusion"]["timesteps"]
    betas = cosine_beta_schedule(T).to(device)

    for i, idx in enumerate(indices):
        cond, gt = dataset[idx]
        cond = cond.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        # Reshape if needed
        if cond.ndim == 5:
            B, C, T_cond, H, W = cond.shape
            cond = cond.view(B, C * T_cond, H, W)

        if gt.ndim == 5 and gt.shape[2] == 1:
            gt = gt.view(gt.shape[0], gt.shape[1], gt.shape[3], gt.shape[4])

        # Generate prediction via denoising
        B, _, H, W = gt.shape
        x = torch.randn(B, 1, H, W, device=device)

        for t in reversed(range(T)):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            mean_pred = model(x, t_tensor, cond)
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            beta_t = betas[t_tensor].view(B, 1, 1, 1)
            x = mean_pred + torch.sqrt(beta_t) * noise

        # Undo normalization
        pred_unnorm = (x * std) + mean
        gt_unnorm = (gt * std) + mean

        pred_np = pred_unnorm[0, 0].cpu().numpy()
        gt_np = gt_unnorm[0, 0].cpu().numpy()

        # Compute metrics
        rmse_val = np.sqrt(np.mean((gt_np - pred_np) ** 2))
        ssim_val = ssim(gt_np, pred_np, data_range=10000)

        # Plot GT
        axs[i, 0].imshow(gt_np, cmap="cividis", vmin=0, vmax=10000)
        axs[i, 0].set_title(f"GT | RMSE: {rmse_val:.1f}")
        axs[i, 0].axis("off")

        # Plot prediction
        axs[i, 1].imshow(pred_np, cmap="cividis", vmin=0, vmax=10000)
        axs[i, 1].set_title(f"Pred | SSIM: {ssim_val:.3f}")
        axs[i, 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ddpm_5pairs_gt_pred.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved GT–Prediction comparison with metrics at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Infer the correct number of input channels from selected vars × time steps
    selected_vars = config["variables"]["selected_condition_vars"]
    angle_vars = config["variables"]["angle_vars"]
    time_steps = config["variables"]["time_steps"]

    cond_channels = (len(selected_vars) + len(angle_vars)) * time_steps

    model = ConditionalDenoisingUNet(
        in_channels=cond_channels,
        model_channels=config["model"]["channels"],
        out_channels=config["model"]["out_channels"]
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load dataset (with selected condition vars)
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

    # Plot prediction–ground truth pairs
    plot_ddpm_sample_pairs(model, dataset, config, args.output_dir, args.device)
