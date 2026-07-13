import torch
import os
import sys
import numpy as np
import torch.distributed as dist
import logging
import time
import calendar
from datetime import date
from torch.nn.parallel import DistributedDataParallel
# from onescience.models.pangu import Pangu
from pangu_patchsize import PanguPatchSize as Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
from onescience.memory.checkpoint import replace_function
from torch.utils.data import Subset
from torch.cuda.amp import autocast
from distill_utils import (
    get_kd_alpha,
    init_bias_table_from_source,
    init_patch_embed_from_official,
    init_patch_embed_from_source,
    load_teacher_model,
    reset_interface_from_official,
    setup_staged_freezing,
)

# 优先使用 apex FusedAdam（更快），不可用时回退到 torch.optim.AdamW
try:
    from apex import optimizers as _apex_optimizers
    FusedAdam = _apex_optimizers.FusedAdam
    print("✅ 使用 apex.FusedAdam 优化器")
except ImportError:
    FusedAdam = torch.optim.AdamW
    print("⚠️  apex 未安装，回退到 torch.optim.AdamW")


TARGET_LOSS_CHANNELS = {
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "geopotential_850",
    "geopotential_700",
    "geopotential_500",
    "specific_humidity_850",
    "specific_humidity_700",
    "specific_humidity_500",
    "temperature_850",
    "temperature_700",
    "temperature_500",
    "u_component_of_wind_500",
    "v_component_of_wind_500",
}


def parse_int_tuple(value, name, expected_length):
    """Parse comma-separated integer model settings from environment variables."""
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers, got {value!r}") from exc
    if len(parsed) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} integers, got {parsed}"
        )
    return parsed


LATITUDE_WEIGHTED_LOSS = os.environ.get("LATITUDE_WEIGHTED_LOSS", "1") != "0"
LOSS_VERSION = (
    "score_aligned_mse_acc_v1_latitude"
    if LATITUDE_WEIGHTED_LOSS
    else "score_aligned_mse_acc_v1_gridpoint"
)
COMPATIBLE_LOSS_VERSIONS = {LOSS_VERSION}
if LATITUDE_WEIGHTED_LOSS:
    COMPATIBLE_LOSS_VERSIONS.add("score_aligned_mse_acc_v1")


def build_target_channel_indices(channels):
    channel_to_index = {name: index for index, name in enumerate(channels)}
    missing = sorted(TARGET_LOSS_CHANNELS.difference(channel_to_index))
    if missing:
        raise ValueError(f"Scored channels are missing from the dataset: {missing}")
    return [
        index for index, name in enumerate(channels)
        if name in TARGET_LOSS_CHANNELS
    ]


def build_latitude_weights(height, device, latitude_weighted=True):
    if not latitude_weighted:
        return torch.ones(1, 1, height, 1, device=device, dtype=torch.float32)
    latitudes = torch.linspace(90.0, -90.0, height, device=device, dtype=torch.float32)
    weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
    return weights.view(1, 1, height, 1)


def score_aligned_loss(
    prediction,
    target,
    target_indices,
    latitude_weights,
    mse_weight=0.8,
    acc_weight=0.2,
    auxiliary_weight=0.05,
    eps=1e-6,
):
    """Equal-channel loss for the 15 scored fields plus a weak auxiliary loss."""
    scored_prediction = prediction.index_select(1, target_indices).float()
    scored_target = target.index_select(1, target_indices).float()

    spatial_weight_sum = latitude_weights.sum() * prediction.shape[-1]
    squared_error = (scored_prediction - scored_target).square()
    per_field_mse = (squared_error * latitude_weights).sum(dim=(-2, -1)) / spatial_weight_sum
    scored_mse = per_field_mse.mean()

    pred_mean = (scored_prediction * latitude_weights).sum(dim=(-2, -1), keepdim=True) / spatial_weight_sum
    target_mean = (scored_target * latitude_weights).sum(dim=(-2, -1), keepdim=True) / spatial_weight_sum
    pred_anomaly = scored_prediction - pred_mean
    target_anomaly = scored_target - target_mean
    covariance = (pred_anomaly * target_anomaly * latitude_weights).sum(dim=(-2, -1))
    pred_variance = (pred_anomaly.square() * latitude_weights).sum(dim=(-2, -1))
    target_variance = (target_anomaly.square() * latitude_weights).sum(dim=(-2, -1))
    correlation = covariance / torch.sqrt(pred_variance * target_variance + eps)
    correlation = correlation.clamp(-1.0, 1.0)
    acc_loss = (1.0 - correlation).mean()

    auxiliary_mask = torch.ones(prediction.shape[1], device=prediction.device, dtype=torch.bool)
    auxiliary_mask[target_indices] = False
    if auxiliary_weight > 0.0 and auxiliary_mask.any():
        auxiliary_error = (prediction[:, auxiliary_mask].float() - target[:, auxiliary_mask].float()).square()
        per_aux_field_mse = (auxiliary_error * latitude_weights).sum(dim=(-2, -1)) / spatial_weight_sum
        auxiliary_loss = per_aux_field_mse.mean()
    else:
        auxiliary_loss = prediction.new_zeros((), dtype=torch.float32)

    total = mse_weight * scored_mse + acc_weight * acc_loss + auxiliary_weight * auxiliary_loss
    components = {
        "mse": scored_mse.detach(),
        "acc": correlation.mean().detach(),
        "aux": auxiliary_loss.detach(),
        "field_mse": per_field_mse.mean(dim=0).detach(),
        "field_acc": correlation.mean(dim=0).detach(),
    }
    return total, components


def score_aligned_kd_loss(
    student_prediction,
    teacher_prediction,
    target_indices,
    latitude_weights,
):
    """Distill only the 15 scored fields, with the same per-field reduction as GT MSE."""
    student_scored = student_prediction.index_select(1, target_indices).float()
    teacher_scored = teacher_prediction.index_select(1, target_indices).detach().float()
    spatial_weight_sum = latitude_weights.sum() * student_prediction.shape[-1]
    per_field_mse = (
        (student_scored - teacher_scored).square() * latitude_weights
    ).sum(dim=(-2, -1)) / spatial_weight_sum
    return per_field_mse.mean()

def should_skip_train_batch(batch_idx, epoch, skip_stride):
    return skip_stride > 0 and batch_idx % skip_stride == (epoch % skip_stride)


def count_kept_train_batches(num_batches, epoch, skip_stride, max_batches):
    kept = 0
    for batch_idx in range(num_batches):
        if should_skip_train_batch(batch_idx, epoch, skip_stride):
            continue
        kept += 1
        if kept >= max_batches:
            break
    return kept


def build_balanced_monthly_indices(dataset, years, days_per_month=10, seed=42):
    """Build a deterministic year/month/time-balanced subset without loading samples."""
    years = [int(year) for year in years]
    if not years:
        raise ValueError("Balanced sampling requires at least one training year")
    if days_per_month <= 0:
        return list(range(len(dataset)))

    if hasattr(dataset, "datasets") and len(dataset.datasets) == len(years):
        year_lengths = [len(year_dataset) for year_dataset in dataset.datasets]
    else:
        if len(dataset) % len(years) != 0:
            raise ValueError(
                "Cannot infer per-year dataset boundaries for balanced sampling: "
                f"dataset={len(dataset)}, years={len(years)}"
            )
        samples_per_year = len(dataset) // len(years)
        year_lengths = [samples_per_year] * len(years)

    indices = []
    offset = 0
    for year, year_length in zip(years, year_lengths):
        year_start = date(year, 1, 1)
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            selected_days = sorted(
                {
                    round((slot + 1) * (days_in_month + 1) / (days_per_month + 1))
                    for slot in range(days_per_month)
                }
            )
            if len(selected_days) != days_per_month:
                raise RuntimeError(
                    f"Could not select {days_per_month} unique days for {year}-{month:02d}"
                )
            for day in selected_days:
                day_offset = (date(year, month, day) - year_start).days
                for time_slot in range(4):
                    local_index = day_offset * 4 + time_slot
                    if local_index >= year_length:
                        raise IndexError(
                            "Balanced sample index exceeds year dataset: "
                            f"year={year}, local_index={local_index}, length={year_length}"
                        )
                    indices.append(offset + local_index)
        offset += year_length

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return indices


def build_every_n_days_indices(dataset, years, day_stride=3, seed=42):
    """Select one complete day every ``day_stride`` days, then shuffle globally.

    ERA5 has four six-hourly samples per day.  For every selected calendar day,
    all four time slots are retained.  Year boundaries are taken from the
    concatenated per-year datasets so leap years remain aligned correctly.
    """
    years = [int(year) for year in years]
    if not years:
        raise ValueError("Every-N-days sampling requires at least one training year")
    if day_stride <= 0:
        return list(range(len(dataset)))

    if hasattr(dataset, "datasets") and len(dataset.datasets) == len(years):
        year_lengths = [len(year_dataset) for year_dataset in dataset.datasets]
    else:
        # ERA5 forecasting datasets commonly expose 1459 usable samples per
        # selected year (the final input has no next-step target), including
        # leap years.  This mirrors train288.py: when the concatenated dataset
        # is evenly divisible, use the actual datapipe length rather than the
        # raw calendar length.
        if len(dataset) % len(years) == 0:
            samples_per_year = len(dataset) // len(years)
            year_lengths = [samples_per_year] * len(years)
        else:
            raise ValueError(
                "Cannot infer per-year boundaries for every-N-days sampling: "
                f"dataset={len(dataset)}, years={len(years)}, "
                "and no per-year child datasets are available"
            )

    indices = []
    selected_days_by_year = {}
    offset = 0
    for year, year_length in zip(years, year_lengths):
        calendar_days = 366 if calendar.isleap(year) else 365
        complete_days = min(calendar_days, year_length // 4)
        selected_day_offsets = list(range(0, complete_days, day_stride))
        selected_days_by_year[year] = len(selected_day_offsets)

        for day_offset in selected_day_offsets:
            day_start = day_offset * 4
            for time_slot in range(4):
                local_index = day_start + time_slot
                if local_index >= year_length:
                    raise IndexError(
                        "Every-N-days sample exceeds year dataset: "
                        f"year={year}, local_index={local_index}, length={year_length}"
                    )
                indices.append(offset + local_index)
        offset += year_length

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return indices, selected_days_by_year


def rebuild_dataloader_with_subset(dataloader, indices):
    subset = Subset(dataloader.dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        shuffle=False,
        collate_fn=dataloader.collate_fn,
        persistent_workers=(
            dataloader.persistent_workers if dataloader.num_workers > 0 else False
        ),
    )


def build_scheduler(optimizer, epoch=0, t_max=100):
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, last_epoch=epoch - 1
    )


def build_optimizer(model, backbone_lr=5e-5, interface_lr=2e-4):
    backbone_params = []
    interface_params = []
    for name, param in model.named_parameters():
        param.requires_grad = True
        if any(layer in name for layer in ["layer1", "layer2", "layer3", "layer4"]):
            backbone_params.append(param)
        else:
            interface_params.append(param)

    return FusedAdam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": interface_params, "lr": interface_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=3e-6,
    )

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")

    ## Distributed config init
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0
    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()

    cfg_data = YParams(config_file_path, "datapipe")
    datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized())
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()

    balanced_days_per_month = int(
        os.environ.get(
            "BALANCED_DAYS_PER_MONTH",
            str(getattr(cfg_data.dataset, "balanced_days_per_month", 0)),
        )
    )
    sample_every_n_days = int(
        os.environ.get(
            "SAMPLE_EVERY_N_DAYS",
            str(getattr(cfg_data.dataset, "sample_every_n_days", 0)),
        )
    )
    if sample_every_n_days > 0 and balanced_days_per_month > 0:
        raise ValueError(
            "SAMPLE_EVERY_N_DAYS and BALANCED_DAYS_PER_MONTH cannot both be enabled"
        )

    if sample_every_n_days > 0:
        if train_sampler is not None:
            raise RuntimeError("Every-N-days sampling currently requires single-device training")
        sampled_indices, selected_days_by_year = build_every_n_days_indices(
            train_dataloader.dataset,
            cfg_data.dataset.train_ratio,
            day_stride=sample_every_n_days,
            seed=int(getattr(cfg_data.dataset, "random_seed", 42)),
        )
        train_dataloader = rebuild_dataloader_with_subset(
            train_dataloader, sampled_indices
        )
        if world_rank == 0:
            print(
                "every-N-days training subset: "
                f"years={list(cfg_data.dataset.train_ratio)}, "
                f"stride={sample_every_n_days} day(s), four times/day, "
                f"selected_days={selected_days_by_year}, "
                f"samples={len(sampled_indices)}, shuffled with "
                f"seed={int(getattr(cfg_data.dataset, 'random_seed', 42))}"
            )
    elif balanced_days_per_month > 0:
        if train_sampler is not None:
            raise RuntimeError("Balanced monthly sampling currently requires single-device training")
        balanced_indices = build_balanced_monthly_indices(
            train_dataloader.dataset,
            cfg_data.dataset.train_ratio,
            days_per_month=balanced_days_per_month,
            seed=int(getattr(cfg_data.dataset, "random_seed", 42)),
        )
        train_dataloader = rebuild_dataloader_with_subset(
            train_dataloader, balanced_indices
        )
        if world_rank == 0:
            print(
                "balanced monthly training subset: "
                f"years={len(cfg_data.dataset.train_ratio)}, "
                f"days/month={balanced_days_per_month}, "
                f"four times/day, samples={len(balanced_indices)}"
            )

    target_channel_indices = build_target_channel_indices(cfg_data.dataset.channels)
    target_indices = torch.as_tensor(target_channel_indices, device=local_rank, dtype=torch.long)
    latitude_weights = build_latitude_weights(
        cfg_data.dataset.img_size[0],
        local_rank,
        latitude_weighted=LATITUDE_WEIGHTED_LOSS,
    )
    score_mse_weight = float(os.environ.get("SCORE_MSE_WEIGHT", "0.8"))
    score_acc_weight = float(os.environ.get("SCORE_ACC_WEIGHT", "0.2"))
    auxiliary_loss_weight = float(os.environ.get("AUXILIARY_LOSS_WEIGHT", "0.05"))
    if world_rank == 0:
        print(
            "score-aligned loss: "
            f"15 scored fields equally weighted, "
            f"spatial weighting={'cos(latitude)' if LATITUDE_WEIGHTED_LOSS else 'gridpoint-equal'}, "
            f"MSE={score_mse_weight}, "
            f"spatial ACC proxy={score_acc_weight}, auxiliary MSE={auxiliary_loss_weight}"
        )

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(local_rank)
    surface_mask = surface_mask.unsqueeze(0)  # [1, 3, H, W], 循环中用 expand 动态匹配 batch_size

    ## Model init
    model = Pangu(img_size=cfg_data.dataset.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  num_heads=cfg.num_heads,
                  window_size=cfg.window_size,
                  depth=cfg.depth,
                  ).to(local_rank)
    
    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    run_tag = os.environ.get("RUN_TAG", "").strip()
    adapt_from = os.environ.get("ADAPT_FROM", "").strip()
    prune_from_env = os.environ.get("PRUNE_FROM", "").strip()
    warm_start_from = os.environ.get("WARM_START_FROM", "").strip()
    if adapt_from and prune_from_env:
        raise ValueError("Set only one of ADAPT_FROM or PRUNE_FROM")
    if warm_start_from and (adapt_from or prune_from_env):
        raise ValueError("WARM_START_FROM cannot be combined with ADAPT_FROM/PRUNE_FROM")
    transfer_from = adapt_from or prune_from_env
    transfer_label = "ADAPT_FROM" if adapt_from else "PRUNE_FROM"
    # ADAPT_FROM and WARM_START_FROM keep the legacy 2_10_10 filenames unless
    # RUN_TAG is explicitly provided.  Structural pruning keeps its historical tag.
    if not run_tag and prune_from_env:
        run_tag = "pruned2642"
        if world_rank == 0:
            print(
                f"PRUNE_FROM is set; generated RUN_TAG={run_tag} "
                "to protect the source checkpoint"
            )
    elif not run_tag and (adapt_from or warm_start_from):
        if world_rank == 0:
            print(
                "legacy output filenames enabled: model_2_10_10_{bak,latest}.pth "
                "and {trloss,valoss}_2_10_10.npy"
            )
    if run_tag and not all(char.isalnum() or char in "-_" for char in run_tag):
        raise ValueError("RUN_TAG may contain only letters, digits, '-' and '_")
    artifact_suffix = f"_{run_tag}" if run_tag else ""
    train_loss_file = f"{cfg.checkpoint_dir}/trloss_2_10_10{artifact_suffix}.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss_2_10_10{artifact_suffix}.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    ## Get model params count
    if cfg.world_size == 1:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n\n")
        print("-" * 50)
        print(f"📂 now params is {total_params}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
        print("-" * 50, "\n")

    ## Checkpoint file names
    best_ckpt_name = f"model_2_10_10{artifact_suffix}_bak.pth"
    latest_ckpt_name = f"model_2_10_10{artifact_suffix}_latest.pth"
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, best_ckpt_name)
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, latest_ckpt_name)
    save_interval_sec = int(os.environ.get("SAVE_INTERVAL_SEC", "600"))

    ## Load model weight if there exist well-trained model
    start_epoch = 0
    start_batch = 0
    loaded_resume = False
    resume_ckpt_data = None
    loss_definition_changed = False
    reset_optimizer = os.environ.get("RESET_OPTIMIZER", "0") != "0"
    override_resume_lr = os.environ.get("OVERRIDE_RESUME_LR", "0") != "0"
    # Conservative defaults for adapting an already-trained model to a new patch size.
    backbone_lr = float(os.environ.get("BACKBONE_LR", "2e-6"))
    interface_lr = float(os.environ.get("INTERFACE_LR", "1e-5"))
    kd_enabled = os.environ.get("KD_ENABLED", "0") != "0"
    kd_interface_only = os.environ.get("KD_INTERFACE_ONLY", "1") != "0"
    kd_alpha_start = float(os.environ.get("KD_ALPHA", "0.05"))
    kd_alpha_end = float(os.environ.get("KD_ALPHA_MIN", "0.01"))
    patch_embed_init = os.environ.get("PATCH_EMBED_INIT", "1") != "0"
    reset_interface = os.environ.get("RESET_INTERFACE", "0") != "0"
    freeze_backbone_epochs = int(os.environ.get("FREEZE_BACKBONE_EPOCHS", "0"))
    teacher_model = None

    if reset_interface:
        reset_optimizer = True

    prune_from = transfer_from
    # Auto-resume an existing transfer run unless explicitly restarted.
    force_retransfer = (
        os.environ.get("FORCE_RETRANSFER", "0") != "0"
        or os.environ.get("FORCE_REPRUNE", "0") != "0"
    )
    if prune_from and not force_retransfer and (os.path.exists(latest_ckpt_path) or os.path.exists(best_ckpt_path)):
        if world_rank == 0:
            print("\n" + "-" * 50)
            print(f"existing transfer checkpoint found: {latest_ckpt_path}")
            print("switching to resume mode; use FORCE_RETRANSFER=1 to restart")
            print("-" * 50 + "\n")
        prune_from = ""
    # ---- New data curriculum: inherit model weights only and start fresh progress. ----
    if warm_start_from:
        if not os.path.exists(warm_start_from):
            raise FileNotFoundError(f"WARM_START_FROM checkpoint not found: {warm_start_from}")
        if world_rank == 0:
            print("\n" + "-" * 50)
            print(f"warm start from: {warm_start_from}")
            print("loading model weights only; optimizer/scheduler/progress reset")
            print("-" * 50 + "\n")
        ckpt = torch.load(warm_start_from, map_location="cpu", weights_only=False)
        source_state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(source_state, strict=True)
        start_epoch = 0
        start_batch = 0
        reset_optimizer = True
        best_valid_loss = 1.0e6
        best_loss_epoch = 0
        train_losses = np.empty((0,), dtype=np.float32)
        valid_losses = np.empty((0,), dtype=np.float32)
        if world_rank == 0:
            logger.info(f"loaded model weights from {warm_start_from}")

    # ---- Step 0 (剪枝 warm start): 从全量 checkpoint 做结构剪枝的权重迁移 ----
    elif prune_from:
        if not os.path.exists(prune_from):
            raise FileNotFoundError(f"{transfer_label} checkpoint not found: {prune_from}")
        if world_rank == 0:
            print("\n")
            print("-" * 50)
            print(
                f"model transfer: loading {prune_from} into "
                f"patch_size={list(cfg.patch_size)}, depth={list(cfg.depth)}"
            )
            print("-" * 50 + "\n")
        ckpt = torch.load(prune_from, map_location="cpu", weights_only=False)
        src_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        source_depth = ckpt.get("depth") if isinstance(ckpt, dict) else None
        if adapt_from and source_depth is not None:
            source_depth = tuple(int(value) for value in source_depth)
            target_depth = tuple(int(value) for value in cfg.depth)
            if source_depth != target_depth:
                raise RuntimeError(
                    "ADAPT_FROM changes patch size only and requires matching depth: "
                    f"source={source_depth}, target={target_depth}. "
                    "Use PRUNE_FROM only when structural pruning is intentional."
                )

        model_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        for k, v in src_state.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped_keys.append(k)

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        real_missing = [k for k in missing_keys if "attn_mask" not in k]
        attn_mask_missing = [k for k in missing_keys if "attn_mask" in k]
        structural_missing = [k for k in real_missing if k not in skipped_keys]

        pruned_count = len(src_state) - len(filtered_state) - len(skipped_keys)
        if adapt_from and pruned_count > 0:
            raise RuntimeError(
                "ADAPT_FROM detected parameters absent from the target model "
                f"({pruned_count} keys). Keep source/target depth identical, or use "
                "PRUNE_FROM when pruning is intentional."
            )
        if adapt_from and structural_missing:
            raise RuntimeError(
                "ADAPT_FROM found target parameters absent from the source checkpoint: "
                f"{structural_missing[:20]}. This usually means source/target depth "
                "does not match. ADAPT_FROM must not randomly initialize Transformer blocks."
            )
        if world_rank == 0:
            print(f"✅ 已迁移 {len(filtered_state)} 个参数")
            print(f"✂️  跳过 {pruned_count} 个被剪枝的参数（layer3.block4/block5 不存在于新模型）")
            if skipped_keys:
                print(f"⚠️  {len(skipped_keys)} 个参数因形状不匹配跳过: {skipped_keys[:10]}")
            print(f"   {len(real_missing)} 个参数随机初始化")
            print(f"   attn_mask buffer {len(attn_mask_missing)} 个将自动重建")
            print(f"   optimizer/scheduler 从头初始化，从 epoch 0 开始训练")

        start_epoch = 0
        start_batch = 0

        # patch_size 变更时，用插值初始化不匹配的 patch_embed/recovery conv 权重
        if patch_embed_init and skipped_keys:
            if world_rank == 0:
                print("initializing mismatched patch projections from PRUNE_FROM source")
            init_patch_embed_from_source(
                model,
                prune_from,
                device="cpu",
                log_fn=print if world_rank == 0 else lambda *args: None,
            )

        # earth_position_bias_table 窗口维度插值初始化
        if patch_embed_init:
            if world_rank == 0:
                print("initializing earth_position_bias_table from PRUNE_FROM source")
            init_bias_table_from_source(
                model,
                prune_from,
                device="cpu",
                log_fn=print if world_rank == 0 else lambda *args: None,
            )

        # Patch adaptation / pruning should start from conservative learning rates.
        if "BACKBONE_LR" not in os.environ:
            backbone_lr = 2e-6
        if "INTERFACE_LR" not in os.environ:
            interface_lr = 1e-5

    # ---- Step 1: 断点续训（优先）：2_10_10 checkpoint = 继承全量模型训练后的结果 ----
    elif os.path.exists(latest_ckpt_path) or os.path.exists(best_ckpt_path):
        resume_ckpt = latest_ckpt_path if os.path.exists(latest_ckpt_path) else best_ckpt_path
        if world_rank == 0:
            print("\n")
            print("-" * 50)
            print(f"loading resume checkpoint: {resume_ckpt}")
            print(f'删除此文件可重新从全量模型 warm start')
            print("-" * 50 + "\n")
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        resume_ckpt_data = ckpt
        if world_rank == 0:
            logger.info(f"loaded student weights from {resume_ckpt}")
        checkpoint_loss_version = ckpt.get("loss_version")
        if checkpoint_loss_version not in COMPATIBLE_LOSS_VERSIONS:
            loss_definition_changed = True
            reset_optimizer = True
            if world_rank == 0:
                print(
                    "loss definition changed "
                    f"({checkpoint_loss_version!r} -> {LOSS_VERSION!r}); "
                    "optimizer, scheduler, best loss, and loss history will be reset"
                )
        saved_epoch = int(ckpt.get("epoch", -1))
        if ckpt.get("epoch_complete", True):
            start_epoch = saved_epoch + 1
            start_batch = 0
        else:
            start_epoch = saved_epoch
            start_batch = int(ckpt.get("next_batch_idx", 0))

        if loss_definition_changed and start_batch != 0:
            if world_rank == 0:
                logger.info(
                    f"new loss warm-up: resetting batch offset {start_batch} -> 0 "
                    "while keeping loaded model weights and epoch"
                )
            start_batch = 0

        if reset_optimizer:
            best_valid_loss = 1.0e6
            best_loss_epoch = start_epoch
            train_losses = np.empty((0,), dtype=np.float32)
            valid_losses = np.empty((0,), dtype=np.float32)
        else:
            best_valid_loss = ckpt.get("best_valid_loss", best_valid_loss)
            best_loss_epoch = ckpt.get("best_loss_epoch", best_loss_epoch)

        if not reset_optimizer and os.path.exists(train_loss_file):
            train_losses = np.load(train_loss_file)
        if not reset_optimizer and os.path.exists(valid_loss_file):
            valid_losses = np.load(valid_loss_file)
        loaded_resume = True

        if reset_interface:
            official_path = os.path.join(cfg.checkpoint_dir, "model_bak.pth")
            if not os.path.exists(official_path):
                raise FileNotFoundError(
                    f"RESET_INTERFACE=1 requires official checkpoint: {official_path}"
                )
            if world_rank == 0:
                print("resetting interface from official [2,4,4] checkpoint")
            reset_count = reset_interface_from_official(
                model,
                official_path,
                device="cpu",
                log_fn=print if world_rank == 0 else lambda *args: None,
            )
            if reset_count == 0:
                raise RuntimeError("Interface reset did not copy or initialize any parameters")

        if world_rank == 0:
            print(f"loaded checkpoint weights, resume training from epoch {start_epoch}, batch {start_batch}")
            if reset_optimizer:
                print("new loss/lr mode: loaded model weights and progress; optimizer/scheduler/best loss reset")

    # ---- Step 1 (备): warm start：无 2_10_10 checkpoint 时，从 244 全量权重部分迁移 ----
    elif os.path.exists(os.path.join(cfg.checkpoint_dir, "model_bak.pth")):
        warm_start_path = os.path.join(cfg.checkpoint_dir, "model_bak.pth")
        if world_rank == 0:
            print("\n")
            print("-" * 50)
            print(f"🔥 warm start from 244 checkpoint: {warm_start_path}")
            print("-" * 50 + "\n")
        ckpt = torch.load(warm_start_path, map_location=f'cuda:{local_rank}', weights_only=False)
        src_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # 按形状过滤：patch_size 变更导致部分参数形状不匹配，strict=False 无法跳过
        model_state = model.state_dict()
        filtered_state = {}
        shape_mismatch_keys = []
        for k, v in src_state.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    shape_mismatch_keys.append(k)

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        real_missing = [k for k in missing_keys if "attn_mask" not in k]
        attn_mask_missing = [k for k in missing_keys if "attn_mask" in k]
        if world_rank == 0:
            print(f"✅ 已迁移 {len(filtered_state)} 个参数（QKV/MLP/LayerNorm）")
            if shape_mismatch_keys:
                print(f"⚠️  {len(shape_mismatch_keys)} 个参数因 patch_size 变更跳过（形状不匹配）:")
                print(f"   {shape_mismatch_keys[:10]}")
            print(f"   {len(real_missing)} 个参数随机初始化（patch_embed/recovery/downsample/upsample）")
            print(f"   attn_mask buffer {len(attn_mask_missing)} 个将自动重建")
            if real_missing:
                print(f"   missing keys (前20): {real_missing[:20]}")
            if unexpected_keys:
                print(f"   unexpected keys (前20): {list(unexpected_keys)[:20]}")
            print("   optimizer/scheduler 从头初始化，从 epoch 0 开始训练")
        start_epoch = 0
        start_batch = 0

        if patch_embed_init:
            if world_rank == 0:
                print("initializing mismatched patch projections: source -> target patch_size")
            initialized_count = init_patch_embed_from_source(
                model,
                warm_start_path,
                device="cpu",
                log_fn=print if world_rank == 0 else lambda *args: None,
            )
            if world_rank == 0:
                print(f"initialized {initialized_count} patch projection weights")
            if initialized_count == 0:
                raise RuntimeError(
                    "PATCH_EMBED_INIT=1 did not find compatible projection weights to initialize"
                )

            # earth_position_bias_table 窗口维度插值初始化
            if world_rank == 0:
                print("initializing earth_position_bias_table from source")
            init_bias_table_from_source(
                model,
                warm_start_path,
                device="cpu",
                log_fn=print if world_rank == 0 else lambda *args: None,
            )

    if world_rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"full fine-tune enabled: {total} parameters trainable")
        print(f"optimizer lr groups: backbone(layer1-4)={backbone_lr}, interface={interface_lr}")

    raw_model = model  # 保留原始模型引用，用于 replace_function
    if world_rank == 0:
        print("torch.compile disabled; training will use eager mode")

    ## Optimizer init: low-lr full fine-tuning with separate backbone/interface groups.
    optimizer = build_optimizer(model, backbone_lr=backbone_lr, interface_lr=interface_lr)
    scheduler = build_scheduler(optimizer, t_max=cfg.max_epoch)

    # Freeze is an experiment-level phase, not a per-process phase.  Restarts must
    # not append another FREEZE_BACKBONE_EPOCHS epochs to the warm-up.
    if resume_ckpt_data is not None:
        freeze_until_epoch = int(
            resume_ckpt_data.get(
                "freeze_until_epoch", max(0, freeze_backbone_epochs)
            )
        )
    else:
        freeze_until_epoch = max(0, freeze_backbone_epochs)

    teacher_ckpt = os.environ.get(
        "TEACHER_CKPT", os.path.join(cfg.checkpoint_dir, "model_bak.pth")
    ).strip()
    teacher_patch_size = parse_int_tuple(
        os.environ.get("TEACHER_PATCH_SIZE", "2,4,4"),
        "TEACHER_PATCH_SIZE",
        3,
    )
    teacher_depth = parse_int_tuple(
        os.environ.get("TEACHER_DEPTH", "2,6,6,2"),
        "TEACHER_DEPTH",
        4,
    )
    kd_will_be_active = kd_enabled and (
        not kd_interface_only or start_epoch < freeze_until_epoch
    )
    if kd_enabled and kd_interface_only and freeze_until_epoch <= 0:
        raise ValueError(
            "KD_INTERFACE_ONLY=1 requires FREEZE_BACKBONE_EPOCHS > 0 "
            "for a new experiment"
        )
    if kd_will_be_active:
        if not os.path.exists(teacher_ckpt):
            raise FileNotFoundError(
                f"KD_ENABLED=1 requires teacher checkpoint: {teacher_ckpt}"
            )
        teacher_model = load_teacher_model(
            teacher_ckpt,
            img_size=cfg_data.dataset.img_size,
            device=local_rank,
            depth=teacher_depth,
            patch_size=teacher_patch_size,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            dtype=torch.bfloat16,
            log_fn=print if world_rank == 0 else lambda *args: None,
        )
        if world_rank == 0:
            print(
                f"distillation enabled: alpha {kd_alpha_start} -> {kd_alpha_end}, "
                f"teacher={teacher_ckpt}, patch_size={list(teacher_patch_size)}, "
                f"depth={list(teacher_depth)}, scored fields only, "
                f"interface_only={kd_interface_only}"
            )

    if world_rank == 0 and freeze_until_epoch > start_epoch:
        print(
            f"backbone warm-up freeze: epochs [0, {freeze_until_epoch}); "
            f"current start_epoch={start_epoch}"
        )

    checkpoint_metadata = {
        "patch_size": list(cfg.patch_size),
        "depth": list(cfg.depth),
        "train_years": [int(year) for year in cfg_data.dataset.train_ratio],
        "validation_years": [int(year) for year in cfg_data.dataset.val_ratio],
        "sample_every_n_days": sample_every_n_days,
        "sampling_seed": int(getattr(cfg_data.dataset, "random_seed", 42)),
        "freeze_until_epoch": freeze_until_epoch,
        "source_checkpoint": warm_start_from or transfer_from or None,
        "transfer_mode": (
            "adapt" if adapt_from else "prune" if prune_from_env else "warm_start"
            if warm_start_from else None
        ),
        "teacher_checkpoint": teacher_ckpt if kd_will_be_active else None,
        "teacher_patch_size": list(teacher_patch_size) if kd_will_be_active else None,
        "teacher_depth": list(teacher_depth) if kd_will_be_active else None,
    }

    ## Only restore optimizer/scheduler when explicitly requested.
    if resume_ckpt_data is not None and not reset_optimizer:
        optimizer.load_state_dict(resume_ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt_data["scheduler_state_dict"])
        if override_resume_lr:
            requested_lrs = (backbone_lr, interface_lr)
            if len(optimizer.param_groups) != len(requested_lrs):
                raise RuntimeError(
                    "Cannot override resumed learning rates: expected two optimizer groups, "
                    f"found {len(optimizer.param_groups)}"
                )
            for group, requested_lr in zip(optimizer.param_groups, requested_lrs):
                group["lr"] = requested_lr
                group["initial_lr"] = requested_lr
            scheduler.base_lrs = list(requested_lrs)
        if world_rank == 0:
            print("✅ optimizer/scheduler 状态已恢复")
            if override_resume_lr:
                print(
                    "resumed learning rates overridden: "
                    f"backbone={backbone_lr}, interface={interface_lr}"
                )
            else:
                print(
                    "resumed learning rates kept from checkpoint: "
                    f"backbone={optimizer.param_groups[0]['lr']}, "
                    f"interface={optimizer.param_groups[1]['lr']}"
                )
    elif resume_ckpt_data is not None and world_rank == 0:
        print("optimizer/scheduler initialized fresh for new loss/lr")

    # 续训时重建 DataLoader 跳过已处理 batch，避免逐个加载数据
    # 注意：仅用于续训 epoch，后续 epoch 必须恢复全量 DataLoader
    resume_batch_offset = start_batch
    original_train_dataloader = train_dataloader
    resume_dataloader = None

    if start_batch > 0 and train_sampler is None:
        dataset = train_dataloader.dataset
        skip_samples = start_batch * cfg_data.dataloader.batch_size
        resume_dataloader = rebuild_dataloader_with_subset(
            train_dataloader,
            range(skip_samples, len(dataset)),
        )
        if world_rank == 0:
            print(
                f"DataLoader rebuilt: {start_batch} batches skipped, "
                f"{len(resume_dataloader.dataset)} samples remain"
            )
        start_batch = 0

    ## Distributed model
    if cfg.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    world_rank == 0 and logger.info(f"start training from epoch {start_epoch} ...")

    for epoch in range(start_epoch, cfg.max_epoch):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0
        train_gt_loss = 0
        train_kd_loss = 0
        processed_batches = 0

        setup_staged_freezing(
            raw_model,
            epoch,
            unfreeze_epoch=freeze_until_epoch,
            log_fn=logger.info if world_rank == 0 else lambda *args: None,
        )
        kd_alpha = get_kd_alpha(
            epoch - start_epoch,
            max(1, cfg.max_epoch - start_epoch),
            kd_alpha_start,
            kd_alpha_end,
        )
        kd_active = kd_enabled and (
            not kd_interface_only or epoch < freeze_until_epoch
        )
        if kd_enabled and kd_interface_only and not kd_active and teacher_model is not None:
            teacher_model = None
            torch.cuda.empty_cache()
            if world_rank == 0:
                logger.info("interface warm-up complete: Teacher released and KD disabled")

        # 续训 epoch 用截断 DataLoader，后续 epoch 恢复全量 DataLoader
        if epoch == start_epoch and resume_dataloader is not None:
            current_dataloader = resume_dataloader
            current_offset = resume_batch_offset
        else:
            current_dataloader = original_train_dataloader
            current_offset = 0

        # train_max_batches / train_skip_stride 始终基于全量 DataLoader 计算
        # 0 means use the complete training set, matching the stable 288 workflow.
        train_max_batches = int(os.environ.get("TRAIN_MAX_BATCHES", "0"))
        if epoch < freeze_until_epoch:
            train_max_batches = int(
                os.environ.get("WARMUP_MAX_BATCHES", "100")
            )
        full_train_total = len(original_train_dataloader)
        if train_max_batches > 0 and train_max_batches < full_train_total:
            train_skip_stride = max(2, round(full_train_total / (full_train_total - train_max_batches)))
        else:
            train_skip_stride = 0
            train_max_batches = full_train_total

        already_processed = count_kept_train_batches(
            current_offset, epoch, train_skip_stride, train_max_batches
        )
        remaining_train_batches = max(0, train_max_batches - already_processed)
        if remaining_train_batches <= 0:
            raise RuntimeError(
                "No training batches remain for this epoch: "
                f"full_train_total={full_train_total}, current_offset={current_offset}, "
                f"already_processed={already_processed}, train_max_batches={train_max_batches}. "
                "Check the checkpoint batch offset and dataset configuration."
            )
        accumulation_steps = int(os.environ.get("ACCUMULATION_STEPS", "2"))
        effective_batch_size = cfg_data.dataloader.batch_size * accumulation_steps
        if world_rank == 0:
            logger.info(f"gradient accumulation: {accumulation_steps} steps, effective batch_size={effective_batch_size}")
        start_time = time.time()
        last_interval_save = time.time()
        if epoch == start_epoch and start_batch > 0:
            logger.info(f"skipping first {start_batch} batches to resume...")
        optimizer.zero_grad()
        for j, data in enumerate(current_dataloader):
            if processed_batches >= remaining_train_batches:
                break
            global_batch_idx = current_offset + j
            if epoch == start_epoch and j < start_batch:
                logger.info(f"  skip progress: {j}/{start_batch} batches")
                continue
            if should_skip_train_batch(global_batch_idx, epoch, train_skip_stride):
                continue
            invar = data[0]
            outvar = data[1]
            invar_surface = invar[:, :4, :, :].to(local_rank, dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to(local_rank, dtype=torch.float32)
            invar = torch.concat([invar_surface, surface_mask.expand(invar_surface.shape[0], -1, -1, -1), invar_upper_air], dim=1)
            tar_surface = outvar[:, :4, :, :].to(local_rank, dtype=torch.float32)
            tar_upper_air = outvar[:, 4:, :, :].to(local_rank, dtype=torch.float32)

            with replace_function(raw_model,["layer2", "layer3"],cfg.world_size > 1):
                with autocast(dtype=torch.bfloat16):
                    out_surface, out_upper_air = model(invar)
                    out_upper_air = out_upper_air.reshape(tar_upper_air.shape)
                prediction = torch.concat([out_surface, out_upper_air], dim=1)
                target = torch.concat([tar_surface, tar_upper_air], dim=1)
                raw_loss, loss_components = score_aligned_loss(
                    prediction,
                    target,
                    target_indices,
                    latitude_weights,
                    mse_weight=score_mse_weight,
                    acc_weight=score_acc_weight,
                    auxiliary_weight=auxiliary_loss_weight,
                )
            # 总 loss（按累积步数缩放，使梯度等价于大 batch）
            if kd_active and teacher_model is not None:
                with torch.no_grad():
                    with autocast(dtype=torch.bfloat16):
                        teacher_surface, teacher_upper_air = teacher_model(invar)
                        teacher_upper_air = teacher_upper_air.reshape(tar_upper_air.shape)
                teacher_prediction = torch.concat(
                    [teacher_surface, teacher_upper_air], dim=1
                )
                kd_loss = score_aligned_kd_loss(
                    prediction,
                    teacher_prediction,
                    target_indices,
                    latitude_weights,
                )
            else:
                kd_loss = raw_loss.new_zeros(())

            total_loss = raw_loss + kd_alpha * kd_loss
            loss = total_loss / accumulation_steps
            loss.backward()
            train_loss += total_loss.item()
            train_gt_loss += raw_loss.item()
            train_kd_loss += kd_loss.item()
            processed_batches += 1
            if processed_batches % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch}-{already_processed + processed_batches}/{train_max_batches} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/processed_batches: .02f}s/{effective_batch_size}batch] '
                            f'loss:{train_loss / processed_batches: .04f} '
                            f'gt:{train_gt_loss / processed_batches:.04f} '
                            f'mse:{loss_components["mse"].item():.04f} '
                            f'acc:{loss_components["acc"].item():.04f} '
                            f'aux:{loss_components["aux"].item():.04f}'
                            + (
                                f' kd:{train_kd_loss / processed_batches:.04f}(alpha={kd_alpha:.03f})'
                                if kd_active else ''
                            ))
                if save_interval_sec > 0 and time.time() - last_interval_save >= save_interval_sec:
                    save_checkpoint(
                        model, optimizer, scheduler, best_valid_loss, best_loss_epoch,
                        cfg.checkpoint_dir, latest_ckpt_name,
                        epoch=epoch, next_batch_idx=current_offset + j + 1, epoch_complete=False,
                        metadata=checkpoint_metadata,
                    )
                    logger.info(f"interval checkpoint saved at epoch={epoch}, next_batch={current_offset + j + 1}")
                    last_interval_save = time.time()
            
        # epoch 结束时处理残留的累积梯度
        if processed_batches % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss = train_loss / max(1, processed_batches)

        model.eval()
        valid_loss = 0
        valid_mse = 0
        valid_acc = 0
        valid_aux = 0
        valid_field_mse = np.zeros(len(target_channel_indices), dtype=np.float64)
        valid_field_acc = np.zeros(len(target_channel_indices), dtype=np.float64)
        # 0 evaluates the complete validation year; set VAL_MAX_SAMPLES to cap it.
        val_max_samples = int(os.environ.get("VAL_MAX_SAMPLES", "0"))
        if "VAL_MAX_BATCHES" in os.environ and "VAL_MAX_SAMPLES" not in os.environ:
            val_max_samples = int(os.environ["VAL_MAX_BATCHES"]) * cfg_data.dataloader.batch_size
        val_dataset = val_dataloader.dataset
        val_total_samples = len(val_dataset)
        if val_max_samples > 0 and val_max_samples < val_total_samples:
            val_indices = np.linspace(0, val_total_samples - 1, num=val_max_samples, dtype=np.int64).tolist()
            current_val_dataloader = torch.utils.data.DataLoader(
                Subset(val_dataset, val_indices),
                batch_size=cfg_data.dataloader.batch_size,
                num_workers=val_dataloader.num_workers,
                pin_memory=val_dataloader.pin_memory,
                drop_last=False,
                shuffle=False,
            )
        else:
            val_max_samples = val_total_samples
            current_val_dataloader = val_dataloader
        val_max_batches = len(current_val_dataloader)
        val_processed = 0
        with torch.no_grad():
            start_time = time.time()
            for j, data in enumerate(current_val_dataloader):
                invar = data[0]
                outvar = data[1]
                invar_surface = invar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                invar_upper_air = invar[:, 4:, :, :].to(local_rank, dtype=torch.float32)
                invar = torch.concat([invar_surface, surface_mask.expand(invar_surface.shape[0], -1, -1, -1), invar_upper_air], dim=1)
                tar_surface = outvar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                tar_upper_air = outvar[:, 4:, :, :].to(local_rank, dtype=torch.float32)

                with autocast(dtype=torch.bfloat16):
                    out_surface, out_upper_air = model(invar)
                    out_upper_air = out_upper_air.reshape(tar_upper_air.shape)
                prediction = torch.concat([out_surface, out_upper_air], dim=1)
                target = torch.concat([tar_surface, tar_upper_air], dim=1)
                loss_tensor, loss_components = score_aligned_loss(
                    prediction,
                    target,
                    target_indices,
                    latitude_weights,
                    mse_weight=score_mse_weight,
                    acc_weight=score_acc_weight,
                    auxiliary_weight=auxiliary_loss_weight,
                )
                # 总 loss
                loss = loss_tensor.item()

                if cfg.world_size > 1:
                    loss_tensor = torch.tensor(loss, device=local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                valid_loss += loss
                valid_mse += loss_components["mse"].item()
                valid_acc += loss_components["acc"].item()
                valid_aux += loss_components["aux"].item()
                valid_field_mse += loss_components["field_mse"].cpu().numpy()
                valid_field_acc += loss_components["field_acc"].cpu().numpy()
                val_processed += 1
                if world_rank == 0:
                    seen_samples = min(val_processed * cfg_data.dataloader.batch_size, val_max_samples)
                    logger.info(f'Valid: Epoch {epoch}-{val_processed}/{val_max_batches} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/val_processed: .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'samples:{seen_samples}/{val_max_samples} '
                            f'loss:{valid_loss / val_processed: .04f} '
                            f'mse:{valid_mse / val_processed:.04f} '
                            f'acc:{valid_acc / val_processed:.04f} '
                            f'aux:{valid_aux / val_processed:.04f}')
                
        valid_loss /= max(1, val_processed)
        valid_field_mse /= max(1, val_processed)
        valid_field_acc /= max(1, val_processed)
        if world_rank == 0 and os.environ.get("LOG_CHANNEL_METRICS", "1") != "0":
            logger.info("Validation metrics by scored field:")
            for channel_index, field_mse, field_acc in zip(
                target_channel_indices, valid_field_mse, valid_field_acc
            ):
                logger.info(
                    f"  {cfg_data.dataset.channels[channel_index]}: "
                    f"mse={field_mse:.6f}, acc={field_acc:.6f}"
                )
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            if world_rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler, best_valid_loss, best_loss_epoch,
                    cfg.checkpoint_dir, best_ckpt_name,
                    epoch=epoch, next_batch_idx=0, epoch_complete=True,
                    metadata=checkpoint_metadata,
                )
            is_save_ckp = True

        scheduler.step()

        if world_rank == 0:
            save_checkpoint(
                model, optimizer, scheduler, best_valid_loss, best_loss_epoch,
                cfg.checkpoint_dir, latest_ckpt_name,
                epoch=epoch, next_batch_idx=0, epoch_complete=True,
                metadata=checkpoint_metadata,
            )

        if world_rank == 0:
            logger.info(f"Epoch [{epoch}/{cfg.max_epoch}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Valid Loss: {valid_loss:.4f}, "
                        f"Best loss at Epoch: {best_loss_epoch}"
                        + (", saving checkpoint" if is_save_ckp else "")
                        )
            train_losses = np.append(train_losses, train_loss)
            valid_losses = np.append(valid_losses, valid_loss)

            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)

        if epoch - best_loss_epoch > cfg.patience:
            print(f"Loss has not decrease in {cfg.patience} epochs, stopping training...")
            exit()


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    best_valid_loss,
    best_loss_epoch,
    model_path,
    filename,
    epoch=None,
    next_batch_idx=0,
    epoch_complete=True,
    metadata=None,
):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
        "epoch": epoch,
        "next_batch_idx": next_batch_idx,
        "epoch_complete": epoch_complete,
        "loss_version": LOSS_VERSION,
    }
    if metadata:
        state.update(metadata)
    os.makedirs(model_path, exist_ok=True)
    tmp_path = os.path.join(model_path, filename + ".tmp")
    dst_path = os.path.join(model_path, filename)
    torch.save(state, tmp_path)
    os.replace(tmp_path, dst_path)
    print(f"saved checkpoint: {dst_path}")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
