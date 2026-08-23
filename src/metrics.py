import numpy as np
import torch
from typing import Dict, Any


def masked_mae_torch(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mape_torch(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / (labels + 1e-5))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100.0


def calculate_metrics_numpy(preds: np.ndarray, labels: np.ndarray, null_val: float = 0.0) -> Dict[str, float]:
    mask = (labels > null_val) & (~np.isnan(labels)) & (~np.isnan(preds))
    if not np.any(mask):
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}

    p = preds[mask]
    l = labels[mask]

    mae = float(np.mean(np.abs(p - l)))
    mse = float(np.mean((p - l) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((p - l) / l)) * 100.0)

    ss_res = np.sum((l - p) ** 2)
    ss_tot = np.sum((l - np.mean(l)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }


def evaluate_horizon_metrics(preds: np.ndarray, labels: np.ndarray, horizons: list = [3, 6, 12]) -> Dict[str, Any]:
    # preds: (N_samples, T_out, N_nodes)
    # labels: (N_samples, T_out, N_nodes)
    results = {}
    
    # Per horizon (1-indexed: 3 -> 15 min, 6 -> 30 min, 12 -> 60 min)
    for h in horizons:
        idx = h - 1
        if idx < preds.shape[1]:
            m = calculate_metrics_numpy(preds[:, idx, :], labels[:, idx, :])
            results[f"horizon_{h}_{h*5}min"] = m

    # Overall average across all prediction steps
    overall = calculate_metrics_numpy(preds, labels)
    results["overall"] = overall
    return results
