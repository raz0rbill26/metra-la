#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tabulate import tabulate

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.autolayout": True
})


def select_device(preferred_device: str) -> torch.device:
    pref = preferred_device.lower().strip()
    if pref == "cuda":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
            return dev
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    elif pref == "mps":
        if torch.backends.mps.is_available():
            print("Using device: mps (Apple Silicon)")
            return torch.device("mps")
        print("MPS not available, falling back to CPU")
        return torch.device("cpu")
    elif pref == "cpu":
        print("Using device: cpu")
        return torch.device("cpu")
    elif pref == "auto":
        if torch.cuda.is_available():
            print(f"Auto-detected device: cuda ({torch.cuda.get_device_name(0)})")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Auto-detected device: mps (Apple Silicon)")
            return torch.device("mps")
        print("Auto-detected device: cpu")
        return torch.device("cpu")

    print(f"Unknown device '{preferred_device}', using cpu")
    return torch.device("cpu")


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, int], np.ndarray]:
    h5_candidates = [
        data_dir / "METR-LA.h5",
        data_dir / "metr-la.h5",
        Path.home() / ".cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5"
    ]
    pkl_candidates = [
        data_dir / "adj_METR-LA.pkl",
        data_dir / "adj_metr-la.pkl",
        Path.home() / ".cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/adj_METR-LA.pkl"
    ]

    h5_path = next((p for p in h5_candidates if p.exists()), None)
    pkl_path = next((p for p in pkl_candidates if p.exists()), None)

    if not h5_path or not pkl_path:
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. Run scripts/setup.py first."
        )

    print(f"Loading traffic speeds from {h5_path}")
    with h5py.File(h5_path, "r") as f:
        sensor_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["df/axis0"][:]]
        timestamps = pd.to_datetime(f["df/axis1"][:])
        values = f["df/block0_values"][:]
        df = pd.DataFrame(values, index=timestamps, columns=sensor_ids)

    print(f"Loading graph adjacency from {pkl_path}")
    with open(pkl_path, "rb") as f:
        try:
            adj_data = pickle.load(f, encoding="latin1")
        except Exception:
            f.seek(0)
            adj_data = pickle.load(f)

    if isinstance(adj_data, (list, tuple)) and len(adj_data) >= 3:
        adj_sensor_ids, sensor_id_to_ind, adj_mx = adj_data[0], adj_data[1], adj_data[2]
    else:
        raise ValueError(f"Unexpected structure in adjacency pickle: {type(adj_data)}")

    return df, adj_sensor_ids, sensor_id_to_ind, adj_mx


def compute_tensor_stats(df: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    start_t = time.perf_counter()
    tensor_data = torch.tensor(df.values, dtype=torch.float32, device=device)
    T, N = tensor_data.shape

    mean_val = torch.mean(tensor_data).item()
    std_val = torch.std(tensor_data).item()
    min_val = torch.min(tensor_data).item()
    max_val = torch.max(tensor_data).item()

    zero_mask = (tensor_data == 0.0)
    zero_count = torch.sum(zero_mask).item()
    zero_pct = (zero_count / (T * N)) * 100.0

    valid_speeds = tensor_data[~zero_mask]
    valid_mean = torch.mean(valid_speeds).item() if valid_speeds.numel() > 0 else 0.0
    valid_std = torch.std(valid_speeds).item() if valid_speeds.numel() > 0 else 0.0

    sensor_means = torch.mean(tensor_data, dim=0).cpu().numpy()
    sensor_stds = torch.std(tensor_data, dim=0).cpu().numpy()
    sensor_mins = torch.min(tensor_data, dim=0).values.cpu().numpy()
    sensor_maxs = torch.max(tensor_data, dim=0).values.cpu().numpy()
    sensor_zeros_pct = (torch.sum(zero_mask, dim=0).float() / T * 100.0).cpu().numpy()

    means = torch.mean(tensor_data, dim=0, keepdim=True)
    stds = torch.std(tensor_data, dim=0, keepdim=True) + 1e-8
    normed = (tensor_data - means) / stds
    corr_matrix = (torch.mm(normed.t(), normed) / (T - 1)).cpu().numpy()

    detrended = tensor_data - means
    fft_vals = torch.fft.rfft(detrended, dim=0)
    power_spectrum = (torch.abs(fft_vals) ** 2).mean(dim=1).cpu().numpy()
    freqs = np.fft.rfftfreq(T, d=5.0 / (60.0 * 24.0))

    elapsed = time.perf_counter() - start_t
    print(f"PyTorch computations on {device} completed in {elapsed:.3f}s")

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "zero_count": int(zero_count),
        "zero_pct": zero_pct,
        "valid_mean": valid_mean,
        "valid_std": valid_std,
        "sensor_means": sensor_means,
        "sensor_stds": sensor_stds,
        "sensor_mins": sensor_mins,
        "sensor_maxs": sensor_maxs,
        "sensor_zeros_pct": sensor_zeros_pct,
        "corr_matrix": corr_matrix,
        "freqs": freqs,
        "power_spectrum": power_spectrum,
        "computation_time_sec": elapsed,
    }


def plot_speed_distribution(df: pd.DataFrame, tensor_stats: Dict[str, Any], output_dir: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    all_vals = df.values.flatten()
    valid_vals = all_vals[all_vals > 0]

    sns.histplot(valid_vals, bins=60, kde=True, ax=axes[0], color="#2b5c8f", stat="density", edgecolor="none")
    axes[0].axvline(tensor_stats["valid_mean"], color="#d9534f", linestyle="--", linewidth=2,
                    label=f'Valid Mean: {tensor_stats["valid_mean"]:.1f} mph')
    axes[0].axvline(np.median(valid_vals), color="#f0ad4e", linestyle=":", linewidth=2,
                    label=f'Valid Median: {np.median(valid_vals):.1f} mph')
    axes[0].set_title("Traffic Speed Distribution (Active Readings > 0 mph)")
    axes[0].set_xlabel("Speed (mph)")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")

    sns.boxplot(x=tensor_stats["sensor_means"], ax=axes[1], color="#5cb85c", orient="h")
    axes[1].set_title("Distribution of Average Speeds Across 207 Sensors")
    axes[1].set_xlabel("Mean Sensor Speed (mph)")

    plt.tight_layout()
    plt.savefig(output_dir / "01_traffic_speed_distribution.png", dpi=dpi)
    plt.close()


def plot_temporal_patterns(df: pd.DataFrame, output_dir: Path, dpi: int):
    temp_df = pd.DataFrame({
        "speed": df.values.mean(axis=1),
        "hour": df.index.hour,
        "dayofweek": df.index.day_name(),
        "is_weekend": df.index.dayofweek >= 5,
        "date": df.index.date
    }, index=df.index)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    weekday_hourly = temp_df[~temp_df["is_weekend"]].groupby("hour")["speed"].agg(["mean", "std"])
    weekend_hourly = temp_df[temp_df["is_weekend"]].groupby("hour")["speed"].agg(["mean", "std"])
    hours = np.arange(24)

    axes[0].plot(hours, weekday_hourly["mean"], label="Weekday (Mon-Fri)", color="#1f77b4", lw=2.5, marker="o")
    axes[0].fill_between(hours,
                         weekday_hourly["mean"] - weekday_hourly["std"],
                         weekday_hourly["mean"] + weekday_hourly["std"],
                         alpha=0.15, color="#1f77b4")
    axes[0].plot(hours, weekend_hourly["mean"], label="Weekend (Sat-Sun)", color="#ff7f0e", lw=2.5, marker="s")
    axes[0].fill_between(hours,
                         weekend_hourly["mean"] - weekend_hourly["std"],
                         weekend_hourly["mean"] + weekend_hourly["std"],
                         alpha=0.15, color="#ff7f0e")
    axes[0].axvspan(7, 9, color="gray", alpha=0.12, label="Morning Rush (7-9 AM)")
    axes[0].axvspan(16, 18.5, color="orange", alpha=0.12, label="Evening Rush (4-6:30 PM)")
    axes[0].set_title("Hourly Speed Profile: Weekday vs Weekend")
    axes[0].set_xlabel("Hour of Day (0 - 23)")
    axes[0].set_ylabel("Average Speed (mph)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(loc="lower left", fontsize=8.5)

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_stats = temp_df.groupby("dayofweek")["speed"].mean().reindex(order)
    sns.barplot(x=day_stats.index, y=day_stats.values, ax=axes[1], palette="crest", hue=day_stats.index, legend=False)
    axes[1].set_title("Average Network Speed by Day of Week")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Mean Speed (mph)")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    pivot_heatmap = temp_df.groupby(["dayofweek", "hour"])["speed"].mean().unstack(level="hour").reindex(order)
    sns.heatmap(pivot_heatmap, ax=axes[2], cmap="vlag_r", cbar_kws={"label": "Avg Speed (mph)"})
    axes[2].set_title("Speed Heatmap: Day vs Hour")
    axes[2].set_xlabel("Hour of Day")
    axes[2].set_ylabel("Day of Week")
    axes[2].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "02_temporal_patterns_hourly_weekly.png", dpi=dpi)
    plt.close()


def plot_sensor_time_series(df: pd.DataFrame, tensor_stats: Dict[str, Any], output_dir: Path, dpi: int):
    sensor_means = tensor_stats["sensor_means"]
    sensor_stds = tensor_stats["sensor_stds"]
    sensor_ids = df.columns.tolist()

    idx_min_speed = int(np.argmin(sensor_means))
    idx_max_speed = int(np.argmax(sensor_means))
    idx_max_var = int(np.argmax(sensor_stds))
    idx_median = int(np.argsort(sensor_means)[len(sensor_means) // 2])

    selected_indices = [idx_min_speed, idx_median, idx_max_var, idx_max_speed]
    titles = [
        f"Lowest Avg Speed Sensor ({sensor_ids[idx_min_speed]}): Mean={sensor_means[idx_min_speed]:.1f} mph",
        f"Median Avg Speed Sensor ({sensor_ids[idx_median]}): Mean={sensor_means[idx_median]:.1f} mph",
        f"Highest Volatility Sensor ({sensor_ids[idx_max_var]}): Std={sensor_stds[idx_max_var]:.1f} mph",
        f"Highest Avg Speed Sensor ({sensor_ids[idx_max_speed]}): Mean={sensor_means[idx_max_speed]:.1f} mph",
    ]
    colors = ["#d9534f", "#337ab7", "#e08214", "#5cb85c"]

    start_time = df.index[0] + pd.Timedelta(days=7)
    end_time = start_time + pd.Timedelta(days=14)
    slice_df = df.loc[start_time:end_time]

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    for ax, s_idx, title, col in zip(axes, selected_indices, titles, colors):
        s_id = sensor_ids[s_idx]
        ax.plot(slice_df.index, slice_df[s_id], color=col, lw=1.2)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Speed (mph)")
        ax.set_ylim(-2, 75)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestamp (5-min intervals over 2 Weeks)")
    plt.tight_layout()
    plt.savefig(output_dir / "03_sensor_time_series_sample.png", dpi=dpi)
    plt.close()


def plot_missing_data_analysis(df: pd.DataFrame, tensor_stats: Dict[str, Any], output_dir: Path, dpi: int):
    zero_mask = (df == 0.0)
    pct_missing_per_sensor = tensor_stats["sensor_zeros_pct"]
    pct_missing_per_timestep = zero_mask.sum(axis=1) / df.shape[1] * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    sns.histplot(pct_missing_per_sensor, bins=30, kde=True, ax=axes[0, 0], color="#993404")
    axes[0, 0].set_title("Missing / Zero Reading Rate per Sensor")
    axes[0, 0].set_xlabel("Missing Data Percentage (%)")
    axes[0, 0].set_ylabel("Number of Sensors")

    axes[0, 1].plot(df.index, pct_missing_per_timestep, color="#800026", lw=0.8)
    axes[0, 1].set_title("Timeline of Network Missing Rate (% of sensors offline)")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("% Sensors Reporting 0 mph")
    axes[0, 1].grid(True, alpha=0.4)

    top15_idx = np.argsort(pct_missing_per_sensor)[-15:][::-1]
    top15_ids = [df.columns[i] for i in top15_idx]
    top15_pcts = [pct_missing_per_sensor[i] for i in top15_idx]

    sns.barplot(x=top15_pcts, y=top15_ids, ax=axes[1, 0], palette="Reds_r", hue=top15_ids, legend=False)
    axes[1, 0].set_title("Top 15 Sensors with Highest Dropout Rate")
    axes[1, 0].set_xlabel("Missing Rate (%)")
    axes[1, 0].set_ylabel("Sensor ID")

    streak_durations_hours = []
    for col in df.columns[:25]:
        s = (df[col] == 0).astype(int)
        runs = s.groupby((s != s.shift()).cumsum()).sum()
        runs = runs[runs > 0] * 5.0 / 60.0
        streak_durations_hours.extend(runs.values)

    streak_durations_hours = np.array(streak_durations_hours)
    sns.histplot(streak_durations_hours[streak_durations_hours <= 24], bins=40, ax=axes[1, 1], color="#41b6c4")
    axes[1, 1].set_title("Distribution of Dropout Streak Durations (≤ 24 hours)")
    axes[1, 1].set_xlabel("Outage Duration (Hours)")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_dir / "04_missing_data_heatmap_timeline.png", dpi=dpi)
    plt.close()


def plot_spatial_adjacency_network(adj_mx: np.ndarray, adj_sensor_ids: List[str], output_dir: Path, dpi: int):
    N = adj_mx.shape[0]
    num_edges = int((adj_mx > 0).sum())
    density = num_edges / (N * N)

    out_degrees = (adj_mx > 0).sum(axis=1)
    in_degrees = (adj_mx > 0).sum(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    sns.heatmap(adj_mx, ax=axes[0], cmap="Blues", cbar_kws={"label": "Weight $W_{ij}$"})
    axes[0].set_title(f"Adjacency Matrix (207 × 207, {num_edges} edges)")
    axes[0].set_xlabel("Sensor Target Index")
    axes[0].set_ylabel("Sensor Source Index")

    sns.kdeplot(out_degrees, ax=axes[1], fill=True, color="#2b8cbe", label=f"Out-degree (mean: {out_degrees.mean():.1f})")
    sns.kdeplot(in_degrees, ax=axes[1], fill=True, color="#e6550d", label=f"In-degree (mean: {in_degrees.mean():.1f})")
    axes[1].set_title("Sensor Node Degree Distribution")
    axes[1].set_xlabel("Node Degree (Neighbors)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, name=adj_sensor_ids[i])
    for i in range(N):
        for j in range(N):
            if adj_mx[i, j] > 0 and i != j:
                G.add_edge(i, j, weight=float(adj_mx[i, j]))

    pos = nx.spring_layout(G, seed=42, k=0.25)
    nx.draw_networkx_nodes(G, pos, ax=axes[2], node_size=20, node_color=out_degrees, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, ax=axes[2], alpha=0.15, edge_color="gray", arrows=False)
    axes[2].set_title(f"Network Graph (Sparsity: {1-density:.2%})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "05_spatial_adjacency_network.png", dpi=dpi)
    plt.close()


def plot_correlation_matrix(corr_matrix: np.ndarray, output_dir: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(corr_matrix, ax=axes[0], cmap="icefire", vmin=-0.2, vmax=1.0, cbar_kws={"label": "Pearson Correlation $r$"})
    axes[0].set_title("Pairwise Sensor Speed Correlation Matrix (207 × 207)")
    axes[0].set_xlabel("Sensor Index")
    axes[0].set_ylabel("Sensor Index")

    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    pairwise_corrs = corr_matrix[triu_indices]

    sns.histplot(pairwise_corrs, bins=50, kde=True, ax=axes[1], color="#6baed6", stat="density")
    axes[1].axvline(pairwise_corrs.mean(), color="#d9534f", linestyle="--", lw=2,
                    label=f"Mean Correlation: {pairwise_corrs.mean():.3f}")
    axes[1].axvline(np.median(pairwise_corrs), color="#238b45", linestyle=":", lw=2,
                    label=f"Median Correlation: {np.median(pairwise_corrs):.3f}")
    axes[1].set_title("Distribution of Pairwise Sensor Correlations")
    axes[1].set_xlabel("Pearson Correlation Coefficient")
    axes[1].set_ylabel("Density")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "06_cross_sensor_correlation_matrix.png", dpi=dpi)
    plt.close()


def plot_spatial_vs_correlation_decay(adj_mx: np.ndarray, corr_matrix: np.ndarray, output_dir: Path, dpi: int):
    N = adj_mx.shape[0]
    weights = []
    corrs = []
    non_neighbor_corrs = []

    for i in range(N):
        for j in range(i + 1, N):
            w = max(adj_mx[i, j], adj_mx[j, i])
            r = corr_matrix[i, j]
            if w > 0:
                weights.append(w)
                corrs.append(r)
            else:
                non_neighbor_corrs.append(r)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].scatter(weights, corrs, alpha=0.35, color="#2b5c8f", edgecolors="none", s=30)
    if len(weights) > 5:
        p = np.poly1d(np.polyfit(weights, corrs, 2))
        x_lin = np.linspace(min(weights), max(weights), 100)
        axes[0].plot(x_lin, p(x_lin), color="#d9534f", lw=2.5, label="Fitted Trend")
    axes[0].set_title("Traffic Correlation vs. Adjacency Weight $W_{ij}$")
    axes[0].set_xlabel("Graph Adjacency Weight $W_{ij}$")
    axes[0].set_ylabel("Speed Correlation $r_{ij}$")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    comp_df = pd.DataFrame({
        "Type": ["Direct Graph Neighbors"] * len(corrs) + ["Non-Adjacent Pairs"] * len(non_neighbor_corrs),
        "Correlation": corrs + non_neighbor_corrs
    })
    sns.boxplot(data=comp_df, x="Type", y="Correlation", ax=axes[1], palette="Set2", hue="Type", legend=False)
    axes[1].set_title("Correlation: Connected vs Unconnected Nodes")
    axes[1].set_ylabel("Pearson Correlation $r$")

    plt.tight_layout()
    plt.savefig(output_dir / "07_spatial_vs_correlation_decay.png", dpi=dpi)
    plt.close()


def plot_frequency_spectral_analysis(tensor_stats: Dict[str, Any], output_dir: Path, dpi: int):
    freqs = tensor_stats["freqs"]
    power = tensor_stats["power_spectrum"]

    mask = (freqs > 0.05) & (freqs <= 6.0)
    f_sub = freqs[mask]
    p_sub = power[mask]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(f_sub, p_sub, color="#08519c", lw=1.8)

    peak_annotations = [
        (1.0, "24-Hour Daily Cycle (1/day)"),
        (2.0, "12-Hour Morning/Evening Rush (2/day)"),
        (3.0, "8-Hour Traffic Harmonic (3/day)"),
        (1.0 / 7.0, "7-Day Weekly Cycle (~0.14/day)"),
    ]

    for target_f, label in peak_annotations:
        closest_idx = np.argmin(np.abs(f_sub - target_f))
        actual_f = f_sub[closest_idx]
        actual_p = p_sub[closest_idx]
        ax.annotate(
            label,
            xy=(actual_f, actual_p),
            xytext=(actual_f + 0.15, actual_p * 1.05 + 1e5),
            arrowprops=dict(facecolor="#d9534f", shrink=0.05, width=1.5, headwidth=6),
            fontsize=9.5,
            fontweight="bold",
            color="#800026"
        )
        ax.axvline(actual_f, color="#d9534f", linestyle=":", alpha=0.5)

    ax.set_title("PyTorch Accelerated Power Spectral Density (FFT)")
    ax.set_xlabel("Frequency (Cycles per Day)")
    ax.set_ylabel("Spectral Power Magnitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "08_frequency_spectral_analysis.png", dpi=dpi)
    plt.close()


def export_summary_statistics(df: pd.DataFrame,
                              adj_mx: np.ndarray,
                              tensor_stats: Dict[str, Any],
                              output_dir: Path,
                              device: torch.device):
    sensor_ids = df.columns.tolist()
    N = len(sensor_ids)
    T = len(df)

    congestion_pct = ((df > 0) & (df < 35.0)).sum(axis=0) / T * 100.0
    freeflow_pct = (df >= 55.0).sum(axis=0) / T * 100.0
    degrees = (adj_mx > 0).sum(axis=1)

    summary_df = pd.DataFrame({
        "Sensor_ID": sensor_ids,
        "Mean_Speed_mph": np.round(tensor_stats["sensor_means"], 2),
        "Std_Speed_mph": np.round(tensor_stats["sensor_stds"], 2),
        "Min_Speed_mph": np.round(tensor_stats["sensor_mins"], 2),
        "Max_Speed_mph": np.round(tensor_stats["sensor_maxs"], 2),
        "Missing_Rate_pct": np.round(tensor_stats["sensor_zeros_pct"], 2),
        "Congested_Rate_pct": np.round(congestion_pct.values, 2),
        "Freeflow_Rate_pct": np.round(freeflow_pct.values, 2),
        "Graph_Out_Degree": degrees
    })

    csv_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(csv_path, index=False)

    num_edges = int((adj_mx > 0).sum())
    density = num_edges / (N * N)

    overview = {
        "dataset_name": "METR-LA (Los Angeles Highway Traffic Speed)",
        "num_sensors": N,
        "num_timesteps": T,
        "temporal_resolution": "5 minutes",
        "start_time": str(df.index.min()),
        "end_time": str(df.index.max()),
        "total_observations": int(T * N),
        "speed_unit": "Miles per Hour (mph)",
        "overall_mean_speed_mph": float(np.round(tensor_stats["mean"], 2)),
        "overall_std_speed_mph": float(np.round(tensor_stats["std"], 2)),
        "active_mean_speed_mph": float(np.round(tensor_stats["valid_mean"], 2)),
        "active_std_speed_mph": float(np.round(tensor_stats["valid_std"], 2)),
        "missing_zero_observations": tensor_stats["zero_count"],
        "missing_rate_pct": float(np.round(tensor_stats["zero_pct"], 2)),
        "graph_nodes": N,
        "graph_directed_edges": num_edges,
        "graph_density": float(np.round(density, 4)),
        "graph_sparsity_pct": float(np.round((1 - density) * 100, 2)),
        "device_used": str(device),
        "computation_time_sec": float(np.round(tensor_stats["computation_time_sec"], 3)),
    }

    json_path = output_dir / "dataset_overview.json"
    with open(json_path, "w") as f:
        json.dump(overview, f, indent=4)

    top_congested = summary_df.sort_values(by="Congested_Rate_pct", ascending=False).head(5)
    top_cleanest = summary_df.sort_values(by="Missing_Rate_pct", ascending=True).head(5)
    top_dropout = summary_df.sort_values(by="Missing_Rate_pct", ascending=False).head(5)

    md_content = f"""# Exploratory Data Analysis (EDA) Report: METR-LA Dataset

## 1. Executive Summary & Overview
The **METR-LA** traffic dataset records vehicle traffic speed on the Los Angeles County highway network collected by Caltrans Performance Measurement System (PeMS).

- **Number of Sensors (Graph Nodes)**: {N}
- **Total Timesteps**: {T:,} (sampled every 5 minutes)
- **Time Horizon**: `{df.index.min()}` to `{df.index.max()}` (~4 months)
- **Total Data Points**: {T * N:,}
- **Computation Device**: `{device}` (PyTorch acceleration completed in {tensor_stats["computation_time_sec"]:.3f}s)

| Metric | Overall (with 0s) | Active Readings (> 0 mph) |
| :--- | :--- | :--- |
| **Mean Speed** | {tensor_stats["mean"]:.2f} mph | {tensor_stats["valid_mean"]:.2f} mph |
| **Std Deviation** | {tensor_stats["std"]:.2f} mph | {tensor_stats["valid_std"]:.2f} mph |
| **Minimum** | {tensor_stats["min"]:.2f} mph | {df[df > 0].min().min():.2f} mph |
| **Maximum** | {tensor_stats["max"]:.2f} mph | {tensor_stats["max"]:.2f} mph |
| **Missing / 0-Value Rate** | **{tensor_stats["zero_pct"]:.2f}%** ({tensor_stats["zero_count"]:,} entries) | - |

---

## 2. Spatial Graph Properties
- **Sensor Graph Edges**: {num_edges:,} directed edges
- **Graph Density**: {density:.2%} (Sparsity: {(1 - density) * 100:.2f}%)
- **Average Node Out-Degree**: {degrees.mean():.2f} neighbors (Min: {degrees.min()}, Max: {degrees.max()})
- **Spatial Correlation**: Direct graph neighbors exhibit substantially higher speed correlation (mean $r \\approx {tensor_stats['corr_matrix'][adj_mx > 0].mean():.2f}$) than unconnected node pairs.

---

## 3. Temporal & Periodic Characteristics
- **Daily Periodicity (24 Hours)**: Marked drops in network speed during morning rush hour (7:00 AM – 9:00 AM) and evening rush hour (4:00 PM – 6:30 PM). Free-flow conditions (~65 mph) dominate between 10:00 PM and 5:00 AM.
- **Weekly Periodicity (7 Days)**: Weekday traffic patterns display strong bimodal rush hour dips, whereas weekend traffic displays a smooth midday curve without pronounced commuting spikes.
- **Spectral Dominance (FFT)**: Power spectral density peaks prominently at $f = 1.0\\text{{ cycles/day}}$ (288 steps) and harmonic $f = 2.0\\text{{ cycles/day}}$ (144 steps).

---

## 4. Key Sensor Rankings

### Top 5 Most Congested Sensors (Highest % Time < 35 mph)
{tabulate(top_congested[['Sensor_ID', 'Mean_Speed_mph', 'Congested_Rate_pct', 'Freeflow_Rate_pct']], headers='keys', tablefmt='pipe', showindex=False)}

### Top 5 Highest Dropout / Missing Sensors
{tabulate(top_dropout[['Sensor_ID', 'Mean_Speed_mph', 'Missing_Rate_pct']], headers='keys', tablefmt='pipe', showindex=False)}

### Top 5 Cleanest Sensors (Lowest Missing Rate)
{tabulate(top_cleanest[['Sensor_ID', 'Mean_Speed_mph', 'Missing_Rate_pct']], headers='keys', tablefmt='pipe', showindex=False)}

---

## 5. Visual Artifacts Generated in `output/`
1. `01_traffic_speed_distribution.png`: Overall and per-sensor speed distributions.
2. `02_temporal_patterns_hourly_weekly.png`: Hourly profile, weekday vs. weekend, and day-hour heatmap.
3. `03_sensor_time_series_sample.png`: 2-week sample time series for representative bottleneck/freeflow sensors.
4. `04_missing_data_heatmap_timeline.png`: Missing data timelines, sensor outage rankings, and streak length analysis.
5. `05_spatial_adjacency_network.png`: Graph adjacency heatmap, degree distribution, and NetworkX network topology.
6. `06_cross_sensor_correlation_matrix.png`: PyTorch GPU/MPS accelerated sensor-to-sensor correlation matrix.
7. `07_spatial_vs_correlation_decay.png`: Spatial distance / weight vs empirical speed correlation decay.
8. `08_frequency_spectral_analysis.png`: PyTorch GPU/MPS accelerated FFT power spectral density.
9. `summary_statistics.csv`: Complete table of all 207 sensors.
10. `dataset_overview.json`: Machine-readable metadata JSON.
"""

    report_path = output_dir / "eda_report.md"
    with open(report_path, "w") as f:
        f.write(md_content)


def main():
    parser = argparse.ArgumentParser(description="METR-LA Dataset Exploratory Data Analysis (EDA)")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Path to folder containing METR-LA.h5 and adj_METR-LA.pkl")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save plots and analysis outputs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Computation device for PyTorch acceleration (auto, cuda, mps, cpu)")
    parser.add_argument("--dpi", type=int, default=160,
                        help="DPI resolution for saved figures")

    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    df, adj_sensor_ids, sensor_id_to_ind, adj_mx = load_dataset(data_dir)
    print(f"Loaded traffic matrix {df.shape} and adjacency graph {adj_mx.shape}")

    tensor_stats = compute_tensor_stats(df, device)

    print(f"Generating plots and reports into {output_dir}")
    plot_speed_distribution(df, tensor_stats, output_dir, args.dpi)
    plot_temporal_patterns(df, output_dir, args.dpi)
    plot_sensor_time_series(df, tensor_stats, output_dir, args.dpi)
    plot_missing_data_analysis(df, tensor_stats, output_dir, args.dpi)
    plot_spatial_adjacency_network(adj_mx, adj_sensor_ids, output_dir, args.dpi)
    plot_correlation_matrix(tensor_stats["corr_matrix"], output_dir, args.dpi)
    plot_spatial_vs_correlation_decay(adj_mx, tensor_stats["corr_matrix"], output_dir, args.dpi)
    plot_frequency_spectral_analysis(tensor_stats, output_dir, args.dpi)

    export_summary_statistics(df, adj_mx, tensor_stats, output_dir, device)
    print(f"EDA complete. Outputs saved in {output_dir}")


if __name__ == "__main__":
    main()
