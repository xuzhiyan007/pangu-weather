"""Precompute the teacher (original full model) per-field reference metrics.

方案B (逐通道自适应权重) 和方案C (W-score 代理损失) 都需要原模型在验证集上
的逐通道 RMSE / ACC 作为参考基准。为了让 student/teacher 的比值可直接比较,
这里复用 train.py 中的 `compute_scored_field_metrics`, 保证:
  * 与训练损失完全相同的 (归一化) 空间;
  * 与训练损失完全相同的 cos(lat) 加权;
  * 与训练损失完全相同的 spatial-mean-removed ACC proxy 定义。

输出 teacher_metrics.json:
  {
    "channels":     [15 个评分通道名, 顺序与 target_indices 一致],
    "rmse":         [15 个逐通道 RMSE (per-sample 均值)],
    "acc":          [15 个逐通道 ACC  (per-sample 均值)],
    "num_samples":  统计使用的样本数,
    "latitude_weighted": bool,
    "patch_size":   teacher patch_size,
    "depth":        teacher depth,
    "loss_space":   "normalized"
  }

用法 (在 pangu_weather 目录下运行):
  python precompute_teacher_metrics.py

可选环境变量:
  TEACHER_CKPT           教师权重路径 (默认 <checkpoint_dir>/model_bak.pth)
  TEACHER_METRICS_OUT    输出 json 路径 (默认 ./teacher_metrics.json)
  TEACHER_DEPTH          教师 depth (默认 2,6,6,2)
  TEACHER_PATCH_SIZE     教师 patch_size (默认 2,4,4)
  METRICS_MAX_SAMPLES    统计样本上限 (默认 500, 0 = 全量验证集)
  LATITUDE_WEIGHTED_LOSS 与训练一致 (默认 1); 必须与训练时保持相同取值
"""

import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import Subset
from torch.cuda.amp import autocast

from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
from distill_utils import load_teacher_model

# 复用训练损失的 helper, 保证 teacher 参考指标与训练目标同定义、同空间。
from train import (
    build_target_channel_indices,
    build_latitude_weights,
    compute_scored_field_metrics,
    LATITUDE_WEIGHTED_LOSS,
)


def _parse_int_tuple(text):
    return tuple(int(value) for value in text.split(","))


def main():
    current_path = os.getcwd()
    sys.path.append(current_path)

    device = 0 if torch.cuda.is_available() else "cpu"

    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    val_dataloader, _ = datapipe.val_dataloader()

    # ---- scored channels / latitude weights (与 train.py 完全一致) ----
    target_channel_indices = build_target_channel_indices(cfg_data.dataset.channels)
    target_indices = torch.as_tensor(target_channel_indices, device=device, dtype=torch.long)
    latitude_weights = build_latitude_weights(
        cfg_data.dataset.img_size[0],
        device,
        latitude_weighted=LATITUDE_WEIGHTED_LOSS,
    )
    scored_channel_names = [cfg_data.dataset.channels[i] for i in target_channel_indices]

    # ---- static surface mask (与 train.py 完全一致) ----
    land_mask = torch.from_numpy(
        np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32)
    )
    topography = torch.from_numpy(
        np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32)
    )
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(device)
    surface_mask = surface_mask.unsqueeze(0)

    # ---- teacher model (原模型, patch_size=[2,4,4], depth=[2,6,6,2]) ----
    teacher_ckpt = os.environ.get(
        "TEACHER_CKPT", os.path.join(cfg.checkpoint_dir, "model_bak.pth")
    )
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(
            f"teacher checkpoint not found: {teacher_ckpt}. "
            "Set TEACHER_CKPT or place model_bak.pth in checkpoint_dir."
        )
    teacher_depth = _parse_int_tuple(os.environ.get("TEACHER_DEPTH", "2,6,6,2"))
    teacher_patch_size = _parse_int_tuple(os.environ.get("TEACHER_PATCH_SIZE", "2,4,4"))
    teacher_model = load_teacher_model(
        teacher_ckpt,
        img_size=cfg_data.dataset.img_size,
        device=device,
        depth=teacher_depth,
        patch_size=teacher_patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    # ---- validation subset (与 train.py 验证循环相同的抽样策略) ----
    max_samples = int(os.environ.get("METRICS_MAX_SAMPLES", "500"))
    val_dataset = val_dataloader.dataset
    val_total_samples = len(val_dataset)
    if 0 < max_samples < val_total_samples:
        val_indices = np.linspace(
            0, val_total_samples - 1, num=max_samples, dtype=np.int64
        ).tolist()
        current_val_dataloader = torch.utils.data.DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=cfg_data.dataloader.batch_size,
            num_workers=val_dataloader.num_workers,
            pin_memory=val_dataloader.pin_memory,
            drop_last=False,
            shuffle=False,
        )
    else:
        current_val_dataloader = val_dataloader

    num_fields = len(target_channel_indices)
    rmse_sum = np.zeros(num_fields, dtype=np.float64)
    acc_sum = np.zeros(num_fields, dtype=np.float64)
    sample_count = 0

    print(
        f"precompute teacher metrics: fields={num_fields}, "
        f"latitude_weighted={LATITUDE_WEIGHTED_LOSS}, "
        f"patch_size={teacher_patch_size}, depth={teacher_depth}, "
        f"val_samples={'all' if max_samples <= 0 else max_samples}/{val_total_samples}"
    )

    with torch.no_grad():
        for j, data in enumerate(current_val_dataloader):
            invar = data[0]
            outvar = data[1]
            invar_surface = invar[:, :4, :, :].to(device, dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to(device, dtype=torch.float32)
            invar = torch.concat(
                [
                    invar_surface,
                    surface_mask.expand(invar_surface.shape[0], -1, -1, -1),
                    invar_upper_air,
                ],
                dim=1,
            )
            tar_surface = outvar[:, :4, :, :].to(device, dtype=torch.float32)
            tar_upper_air = outvar[:, 4:, :, :].to(device, dtype=torch.float32)

            with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out_surface, out_upper_air = teacher_model(invar)
                out_upper_air = out_upper_air.reshape(tar_upper_air.shape)
            prediction = torch.concat([out_surface, out_upper_air], dim=1)
            target = torch.concat([tar_surface, tar_upper_air], dim=1)

            scored_prediction = prediction.index_select(1, target_indices).float()
            scored_target = target.index_select(1, target_indices).float()
            _, per_field_rmse, correlation = compute_scored_field_metrics(
                scored_prediction, scored_target, latitude_weights
            )

            batch_size = per_field_rmse.shape[0]
            rmse_sum += per_field_rmse.sum(dim=0).double().cpu().numpy()
            acc_sum += correlation.sum(dim=0).double().cpu().numpy()
            sample_count += batch_size

            if (j + 1) % 20 == 0:
                print(f"  processed {sample_count} samples...")

    if sample_count == 0:
        raise RuntimeError("No validation samples were processed; check the dataset config")

    teacher_rmse = (rmse_sum / sample_count).tolist()
    teacher_acc = (acc_sum / sample_count).tolist()

    output_path = os.environ.get("TEACHER_METRICS_OUT", "./teacher_metrics.json")
    metrics = {
        "channels": scored_channel_names,
        "rmse": teacher_rmse,
        "acc": teacher_acc,
        "num_samples": sample_count,
        "latitude_weighted": bool(LATITUDE_WEIGHTED_LOSS),
        "patch_size": list(teacher_patch_size),
        "depth": list(teacher_depth),
        "loss_space": "normalized",
    }
    with open(output_path, "w") as out_file:
        json.dump(metrics, out_file, indent=2)

    print("\n" + "-" * 60)
    print(f"teacher metrics saved to {output_path} ({sample_count} samples)")
    for name, rmse_value, acc_value in zip(scored_channel_names, teacher_rmse, teacher_acc):
        print(f"  {name:<32s} rmse={rmse_value:.5f}  acc={acc_value:.5f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
