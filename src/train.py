#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tabulate import tabulate
from tqdm import tqdm

from src.dataset import build_traffic_dataloaders, StandardScaler
from src.metrics import masked_mae_torch, evaluate_horizon_metrics
from src.models import TemporalTransformer, Dilated1DCNN, MultiLayerLSTM, SpatioTemporalGNN

DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_LOG_DIR = ROOT_DIR / ".logs"

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True
})


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("traffic_dl")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def select_device(preferred: str, logger: logging.Logger) -> torch.device:
    pref = preferred.lower().strip()
    if pref == "cuda":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
            return dev
        logger.info("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    elif pref == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using device: mps (Apple Silicon Metal Performance Shaders)")
            return torch.device("mps")
        logger.info("MPS not available, falling back to CPU")
        return torch.device("cpu")
    elif pref == "cpu":
        logger.info("Using device: cpu")
        return torch.device("cpu")
    elif pref == "auto":
        if torch.cuda.is_available():
            logger.info(f"Auto-detected device: cuda ({torch.cuda.get_device_name(0)})")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Auto-detected device: mps (Apple Silicon)")
            return torch.device("mps")
        logger.info("Auto-detected device: cpu")
        return torch.device("cpu")
    logger.info(f"Unknown device '{preferred}', falling back to CPU")
    return torch.device("cpu")


def build_model(model_type: str, num_nodes: int, args: argparse.Namespace) -> nn.Module:
    mtype = model_type.lower().strip()
    if mtype in ["temporal_transformer", "transformer"]:
        return TemporalTransformer(
            num_nodes=num_nodes,
            in_features=2,
            in_steps=args.in_steps,
            out_steps=args.out_steps,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            dropout=args.dropout
        )
    elif mtype in ["cnn", "dilated_cnn", "1d_cnn"]:
        return Dilated1DCNN(
            num_nodes=num_nodes,
            in_features=2,
            in_steps=args.in_steps,
            out_steps=args.out_steps,
            channels=args.d_model,
            kernel_size=2,
            dilations=[1, 2, 4, 8],
            dropout=args.dropout
        )
    elif mtype in ["lstm", "multi_layer_lstm", "rnn"]:
        return MultiLayerLSTM(
            num_nodes=num_nodes,
            in_features=2,
            in_steps=args.in_steps,
            out_steps=args.out_steps,
            hidden_dim=args.d_model,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif mtype in ["gnn", "stgcn", "spatio_temporal_gnn"]:
        return SpatioTemporalGNN(
            num_nodes=num_nodes,
            in_features=2,
            in_steps=args.in_steps,
            out_steps=args.out_steps,
            hidden_channels=args.d_model,
            k=3,
            num_blocks=args.num_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: temporal_transformer, cnn, lstm, gnn")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out_scaled = model(x_batch)

        out_real = scaler.inverse_transform(out_scaled)
        y_real = scaler.inverse_transform(y_batch)

        loss = masked_mae_torch(out_real, y_real, null_val=0.0)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Evaluating", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            out_scaled = model(x_batch)
            out_real = scaler.inverse_transform(out_scaled)
            y_real = scaler.inverse_transform(y_batch)

            loss = masked_mae_torch(out_real, y_real, null_val=0.0)
            total_loss += loss.item()

            preds_list.append(out_real.cpu().numpy())
            labels_list.append(y_real.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, preds, labels


# =============================================================================
# Plotting & Visualization Functions
# =============================================================================

def plot_training_curves(history: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="Train Loss (MAE)", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss (MAE)", color="#ff7f0e", lw=2)
    axes[0].set_title("Training and Validation Loss (Masked MAE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (mph)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["learning_rate"], label="Learning Rate", color="#2ca02c", lw=2)
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_horizon_comparison(horizon_metrics: Dict[str, Any], save_path: Path):
    horizons = ["15 min", "30 min", "60 min", "Overall"]
    keys = ["horizon_3_15min", "horizon_6_30min", "horizon_12_60min", "overall"]

    mae_vals = [horizon_metrics[k]["mae"] for k in keys]
    rmse_vals = [horizon_metrics[k]["rmse"] for k in keys]
    mape_vals = [horizon_metrics[k]["mape"] for k in keys]

    x = np.arange(len(horizons))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, mae_vals, width, label="MAE (mph)", color="#3182bd")
    ax.bar(x, rmse_vals, width, label="RMSE (mph)", color="#de2d26")
    ax.bar(x + width, mape_vals, width, label="MAPE (%)", color="#31a354")

    for i in range(len(horizons)):
        ax.text(i - width, mae_vals[i] + 0.1, f"{mae_vals[i]:.2f}", ha="center", fontsize=9)
        ax.text(i, rmse_vals[i] + 0.1, f"{rmse_vals[i]:.2f}", ha="center", fontsize=9)
        ax.text(i + width, mape_vals[i] + 0.1, f"{mape_vals[i]:.1f}%", ha="center", fontsize=9)

    ax.set_title("Multi-Horizon Forecasting Performance (15, 30, 60 Minutes)")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_predictions_vs_actual(preds: np.ndarray, labels: np.ndarray, sensor_ids: list, save_path: Path):
    window = min(288, len(preds))
    time_indices = np.arange(window) * 5.0 / 60.0

    mean_sensor_speeds = labels[:window, 0, :].mean(axis=0)
    idx_min = int(np.argmin(mean_sensor_speeds))
    idx_max = int(np.argmax(mean_sensor_speeds))
    idx_mid = int(np.argsort(mean_sensor_speeds)[len(mean_sensor_speeds) // 2])
    idx_var = int(np.argmax(labels[:window, 0, :].std(axis=0)))

    chosen_indices = [idx_min, idx_mid, idx_var, idx_max]
    titles = [
        f"Bottleneck Sensor ({sensor_ids[idx_min]})",
        f"Median Sensor ({sensor_ids[idx_mid]})",
        f"High-Volatility Sensor ({sensor_ids[idx_var]})",
        f"Free-Flow Sensor ({sensor_ids[idx_max]})",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=True)
    step_cols = [2, 11]
    step_names = ["15-Minute Forecast (Horizon 3)", "60-Minute Forecast (Horizon 12)"]

    for row, (s_idx, s_title) in enumerate(zip(chosen_indices, titles)):
        for col, (step_idx, step_name) in enumerate(zip(step_cols, step_names)):
            ax = axes[row, col]
            act = labels[:window, step_idx, s_idx]
            prd = preds[:window, step_idx, s_idx]

            ax.plot(time_indices, act, label="Actual", color="#252525", lw=1.5)
            ax.plot(time_indices, prd, label="Predicted", color="#e6550d" if col == 0 else "#2b8cbe", lw=1.5, linestyle="--")
            if row == 0:
                ax.set_title(f"{step_name}\n{s_title}")
            else:
                ax.set_title(s_title)
            ax.set_ylabel("Speed (mph)")
            ax.set_ylim(-2, 75)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=8.5)

    axes[-1, 0].set_xlabel("Time (Hours over 24-hr Test Period)")
    axes[-1, 1].set_xlabel("Time (Hours over 24-hr Test Period)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_sensor_error_distribution(preds: np.ndarray, labels: np.ndarray, sensor_ids: list, save_path: Path):
    mask = labels > 0
    diff = np.abs(preds - labels)
    sensor_mae = np.sum(diff * mask, axis=(0, 1)) / (np.sum(mask, axis=(0, 1)) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(sensor_mae, bins=35, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].axvline(sensor_mae.mean(), color="#d9534f", linestyle="--", lw=2, label=f"Mean: {sensor_mae.mean():.2f} mph")
    axes[0].axvline(np.median(sensor_mae), color="#238b45", linestyle=":", lw=2, label=f"Median: {np.median(sensor_mae):.2f} mph")
    axes[0].set_title("Distribution of Test MAE Across 207 Highway Sensors")
    axes[0].set_xlabel("Sensor Mean Absolute Error (mph)")
    axes[0].set_ylabel("Number of Sensors")
    axes[0].legend()

    ranked_idx = np.argsort(sensor_mae)
    best5_ids = [sensor_ids[i] for i in ranked_idx[:5]]
    best5_mae = [sensor_mae[i] for i in ranked_idx[:5]]
    worst5_ids = [sensor_ids[i] for i in ranked_idx[-5:]][::-1]
    worst5_mae = [sensor_mae[i] for i in ranked_idx[-5:]][::-1]

    comp_ids = best5_ids + worst5_ids
    comp_mae = best5_mae + worst5_mae
    comp_type = ["Top 5 Lowest Error"] * 5 + ["Top 5 Highest Error"] * 5

    df_rank = pd.DataFrame({"Sensor_ID": comp_ids, "MAE": comp_mae, "Category": comp_type})
    sns.barplot(data=df_rank, x="MAE", y="Sensor_ID", hue="Category", ax=axes[1], palette="Set1")
    axes[1].set_title("Extreme Sensor Performance (Top 5 Best vs Worst)")
    axes[1].set_xlabel("MAE (mph)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def run_training_pipeline(args: argparse.Namespace):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run-{args.model}-{timestamp}"

    out_base = Path(args.output_dir).resolve()
    run_dir = out_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_base = Path(args.log_dir).resolve()
    log_base.mkdir(parents=True, exist_ok=True)
    log_file = log_base / f"{run_name}.log"

    logger = setup_logger(log_file)

    cmd_string = f"python {' '.join(sys.argv)}"
    with open(run_dir / "run_command.txt", "w") as f:
        f.write(cmd_string + "\n")

    logger.info("=" * 60)
    logger.info(f"STARTING TRAFFIC FORECASTING RUN: {run_name}")
    logger.info(f"Model Architecture: {args.model}")
    logger.info(f"Command: {cmd_string}")
    logger.info(f"Run Directory: {run_dir}")
    logger.info(f"Log File: {log_file}")
    logger.info("=" * 60)

    device = select_device(args.device, logger)

    logger.info("Building sliding window dataset and dataloaders...")
    data_path = Path(args.data_dir).resolve()
    train_loader, val_loader, test_loader, scaler, sensor_ids = build_traffic_dataloaders(
        data_dir=data_path,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        batch_size=args.batch_size
    )

    logger.info(f"Dataset split: Train batches = {len(train_loader)}, Val batches = {len(val_loader)}, Test batches = {len(test_loader)}")
    logger.info(f"Scaler parameters: Mean = {scaler.mean:.2f}, Std = {scaler.std:.2f}")
    logger.info(f"Total Sensors: {len(sensor_ids)}")

    logger.info(f"Instantiating model: {args.model}")
    model = build_model(args.model, len(sensor_ids), args).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    config = {
        "run_name": run_name,
        "timestamp": timestamp,
        "command": cmd_string,
        "model": args.model,
        "device": str(device),
        "total_params": total_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "in_steps": args.in_steps,
        "out_steps": args.out_steps,
        "scaler": scaler.to_dict(),
        "num_sensors": len(sensor_ids)
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=4)

    best_val_loss = float("inf")
    patience_counter = 0
    history_records = []
    best_checkpoint_path = run_dir / "best_model.pt"

    start_train_time = time.time()
    logger.info("\nStarting training loop...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            scaler=scaler,
            optimizer=optimizer,
            device=device
        )

        val_loss, _, _ = evaluate(
            model=model,
            dataloader=val_loader,
            scaler=scaler,
            device=device
        )

        scheduler.step(val_loss)
        epoch_duration = time.time() - epoch_start

        history_records.append({
            "epoch": epoch,
            "train_loss": float(np.round(train_loss, 4)),
            "val_loss": float(np.round(val_loss, 4)),
            "learning_rate": curr_lr,
            "duration_sec": float(np.round(epoch_duration, 2))
        })

        is_best = val_loss < best_val_loss
        best_marker = "(*)" if is_best else ""
        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} mph | "
            f"Val Loss: {val_loss:.4f} mph | "
            f"LR: {curr_lr:.2e} | "
            f"Time: {epoch_duration:.1f}s {best_marker}"
        )

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience = {args.patience})")
                break

    total_train_duration = time.time() - start_train_time
    logger.info(f"\nTraining completed in {total_train_duration / 60:.2f} minutes.")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f} mph")

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    plot_training_curves(history_df, run_dir / "01_loss_curve.png")

    logger.info("\nLoading best model checkpoint for final Test Set evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_preds, test_labels = evaluate(
        model=model,
        dataloader=test_loader,
        scaler=scaler,
        device=device
    )

    horizon_results = evaluate_horizon_metrics(test_preds, test_labels, horizons=[3, 6, 12])

    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL TEST SET MULTI-HORIZON EVALUATION RESULTS ({args.model.upper()})")
    logger.info("=" * 60)

    table_data = [
        ["15 Minutes (Horizon 3)", f"{horizon_results['horizon_3_15min']['mae']:.2f} mph", f"{horizon_results['horizon_3_15min']['rmse']:.2f} mph", f"{horizon_results['horizon_3_15min']['mape']:.2f}%", f"{horizon_results['horizon_3_15min']['r2']:.4f}"],
        ["30 Minutes (Horizon 6)", f"{horizon_results['horizon_6_30min']['mae']:.2f} mph", f"{horizon_results['horizon_6_30min']['rmse']:.2f} mph", f"{horizon_results['horizon_6_30min']['mape']:.2f}%", f"{horizon_results['horizon_6_30min']['r2']:.4f}"],
        ["60 Minutes (Horizon 12)", f"{horizon_results['horizon_12_60min']['mae']:.2f} mph", f"{horizon_results['horizon_12_60min']['rmse']:.2f} mph", f"{horizon_results['horizon_12_60min']['mape']:.2f}%", f"{horizon_results['horizon_12_60min']['r2']:.4f}"],
        ["Overall Average (12 steps)", f"{horizon_results['overall']['mae']:.2f} mph", f"{horizon_results['overall']['rmse']:.2f} mph", f"{horizon_results['overall']['mape']:.2f}%", f"{horizon_results['overall']['r2']:.4f}"]
    ]
    headers = ["Horizon", "MAE", "RMSE", "MAPE", "R² Score"]
    formatted_table = tabulate(table_data, headers=headers, tablefmt="grid")
    logger.info("\n" + formatted_table)

    with open(run_dir / "metrics_summary.json", "w") as f:
        json.dump(horizon_results, f, indent=4)

    df_metrics = pd.DataFrame(table_data, columns=headers)
    df_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)

    logger.info("\nGenerating evaluation figures...")
    plot_horizon_comparison(horizon_results, run_dir / "02_horizon_metrics_bar.png")
    plot_predictions_vs_actual(test_preds, test_labels, sensor_ids, run_dir / "03_predictions_vs_actual.png")
    plot_sensor_error_distribution(test_preds, test_labels, sensor_ids, run_dir / "04_sensor_error_distribution.png")

    logger.info("=" * 60)
    logger.info(f"RUN COMPLETED SUCCESSFULLY: {args.model}")
    logger.info(f"Artifacts saved in: {run_dir}")
    logger.info(f"Log saved in: {log_file}")
    logger.info("=" * 60)
    return horizon_results, run_dir


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Traffic Forecasting Models on METR-LA")
    parser.add_argument("--model", type=str, default="temporal_transformer",
                        choices=["temporal_transformer", "cnn", "lstm", "gnn"],
                        help="Model architecture to train")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Computation device")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--d-model", type=int, default=64, help="Hidden channel / model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of model layers / blocks")
    parser.add_argument("--d-ff", type=int, default=128, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--in-steps", type=int, default=12, help="Input historical steps (default 12 = 1 hr)")
    parser.add_argument("--out-steps", type=int, default=12, help="Output prediction steps (default 12 = 1 hr)")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output base directory")
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR), help="Log directory")

    args = parser.parse_args()
    run_training_pipeline(args)


if __name__ == "__main__":
    main()
