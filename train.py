import torch
import os
import sys
import numpy as np
import torch.distributed as dist
import logging
import time
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
# from onescience.models.pangu import Pangu
from pangu_patchsize import PanguPatchSize as Pangu
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
from onescience.memory.checkpoint import replace_function
from torch.utils.data import Subset
from torch.cuda.amp import autocast
from distill_utils import (
    init_patch_embed_from_official,
    load_teacher_model,
    compute_kd_loss,
    setup_staged_freezing,
    get_kd_alpha,
    reset_interface_from_official,
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

def build_score_focused_weights(
    channels,
    base_weights,
    target_weight=1.0,
    aux_scale=0.5,
    aux_min=0.03,
    aux_max=0.15,
):
    weights = []
    counts = {"target": 0, "auxiliary": 0}
    for channel, base_weight in zip(channels, base_weights):
        if channel in TARGET_LOSS_CHANNELS:
            weights.append(target_weight)
            counts["target"] += 1
        else:
            weights.append(min(aux_max, max(aux_min, float(base_weight) * aux_scale)))
            counts["auxiliary"] += 1
    return weights, counts


def loss_func(x, y, weights, level_weight=1.0):
    return level_weight * (F.l1_loss(x, y, reduction='none') * weights).mean()


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


def build_scheduler(optimizer, epoch=0):
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, last_epoch=epoch - 1
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

    score_weights, score_weight_counts = build_score_focused_weights(
        cfg_data.dataset.channels,
        cfg_data.dataset.weights,
    )
    surface_weights = torch.as_tensor(score_weights[:4], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)
    pressure_weights = torch.as_tensor(score_weights[4:], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)
    if world_rank == 0:
        print(
            "score-focused loss weights: "
            f"target={score_weight_counts['target']}x1.0, "
            f"auxiliary={score_weight_counts['auxiliary']} scaled from config weights "
            "(clamp 0.03..0.15, scale=0.5)"
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
    train_loss_file = f"{cfg.checkpoint_dir}/trloss_288.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss_288.npy"
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
    best_ckpt_name = "model_288_bak.pth"
    latest_ckpt_name = "model_288_latest.pth"
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, best_ckpt_name)
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, latest_ckpt_name)
    save_interval_sec = int(os.environ.get("SAVE_INTERVAL_SEC", "600"))

    ## Load model weight if there exist well-trained model
    start_epoch = 0
    start_batch = 0
    loaded_resume = False
    resume_ckpt_data = None
    reset_optimizer = os.environ.get("RESET_OPTIMIZER", "0") != "0"
    backbone_lr = float(os.environ.get("BACKBONE_LR", "5e-5"))
    interface_lr = float(os.environ.get("INTERFACE_LR", "2e-4"))

    # ---- 蒸馏 & 分阶段冻结配置 ----
    kd_enabled = os.environ.get("KD_ENABLED", "0") != "0"
    kd_alpha_start = float(os.environ.get("KD_ALPHA", "1.0"))
    kd_alpha_end = float(os.environ.get("KD_ALPHA_MIN", "0.3"))
    unfreeze_epoch = int(os.environ.get("UNFREEZE_EPOCH", "2"))
    surface_level_weight = float(os.environ.get("SURFACE_LEVEL_WEIGHT", "1.0"))
    patch_embed_init = os.environ.get("PATCH_EMBED_INIT", "1") != "0"
    reset_interface = os.environ.get("RESET_INTERFACE", "0") != "0"
    teacher_model = None

    # RESET_INTERFACE 联动：自动冻结 Transformer + 重置 optimizer
    if reset_interface:
        unfreeze_epoch = max(unfreeze_epoch, 999)
        reset_optimizer = True

    # ---- Step 1: 断点续训（优先）：288 checkpoint = 继承全量模型训练后的结果 ----
    if os.path.exists(latest_ckpt_path) or os.path.exists(best_ckpt_path):
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
        saved_epoch = int(ckpt.get("epoch", -1))
        if ckpt.get("epoch_complete", True):
            start_epoch = saved_epoch + 1
            start_batch = 0
        else:
            start_epoch = saved_epoch
            start_batch = int(ckpt.get("next_batch_idx", 0))

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

        # ---- 可选：从官方模型重置接口层（不删除 288 checkpoint）----
        if reset_interface:
            official_path = os.path.join(cfg.checkpoint_dir, "model_bak.pth")
            if os.path.exists(official_path):
                if world_rank == 0:
                    print("\n" + "-" * 50)
                    print("🔄 重置接口层：从官方模型重置 downsample/upsample/patch_embed/recovery")
                reset_interface_from_official(model, official_path, device='cpu', log_fn=print)
                if world_rank == 0:
                    print("-" * 50 + "\n")

        if world_rank == 0:
            print(f"loaded checkpoint weights, resume training from epoch {start_epoch}, batch {start_batch}")
            if reset_optimizer:
                print("new loss/lr mode: loaded model weights and progress; optimizer/scheduler/best loss reset")

    # ---- Step 1 (备): warm start：无 288 checkpoint 时，从 244 全量权重部分迁移 ----
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

        # ---- patch_embed 智能初始化：4×4 → 8×8 聚合 ----
        if patch_embed_init:
            official_path = os.path.join(cfg.checkpoint_dir, "model_bak.pth")
            if os.path.exists(official_path):
                if world_rank == 0:
                    print("\n" + "-" * 50)
                    print("🔧 patch_embed 智能初始化 (4×4 → 8×8 聚合)")
                n = init_patch_embed_from_official(model, official_path, device='cpu', log_fn=print)
                if world_rank == 0:
                    print(f"   初始化 {n} 个 conv 权重")
                    print("-" * 50 + "\n")

    if world_rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"full fine-tune enabled: {total} parameters trainable")
        print(f"optimizer lr groups: backbone(layer1-4)={backbone_lr}, interface={interface_lr}")

    raw_model = model  # 保留原始模型引用，用于 replace_function
    if world_rank == 0:
        print("torch.compile disabled; training will use eager mode")

    ## Optimizer init: low-lr full fine-tuning with separate backbone/interface groups.
    optimizer = build_optimizer(model, backbone_lr=backbone_lr, interface_lr=interface_lr)
    scheduler = build_scheduler(optimizer)

    # ---- 加载 Teacher 模型（蒸馏模式）----
    if kd_enabled:
        teacher_ckpt = os.path.join(cfg.checkpoint_dir, "model_bak.pth")
        teacher_depth = tuple(int(x) for x in os.environ.get("TEACHER_DEPTH", "2,6,6,2").split(","))
        if os.path.exists(teacher_ckpt):
            teacher_model = load_teacher_model(
                teacher_ckpt,
                img_size=cfg_data.dataset.img_size,
                device=local_rank,
                depth=teacher_depth,
                patch_size=(2, 4, 4),
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                log_fn=print if world_rank == 0 else lambda *a: None,
            )
            if world_rank == 0:
                print(f"🎓 蒸馏模式: teacher=[2,4,4] depth={teacher_depth}, "
                      f"KD alpha {kd_alpha_start}→{kd_alpha_end}, unfreeze@epoch {unfreeze_epoch}")
        else:
            if world_rank == 0:
                print(f"⚠️  KD_ENABLED=1 但未找到 teacher checkpoint: {teacher_ckpt}")
            kd_enabled = False

    ## Only restore optimizer/scheduler when explicitly requested.
    if resume_ckpt_data is not None and not reset_optimizer:
        optimizer.load_state_dict(resume_ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt_data["scheduler_state_dict"])
        if world_rank == 0:
            print("✅ optimizer/scheduler 状态已恢复")
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
        subset = Subset(dataset, range(skip_samples, len(dataset)))
        resume_dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=cfg_data.dataloader.batch_size,
            num_workers=train_dataloader.num_workers,
            pin_memory=train_dataloader.pin_memory,
            drop_last=train_dataloader.drop_last,
            shuffle=False,
        )
        if world_rank == 0:
            print(f"DataLoader rebuilt: {start_batch} batches skipped, {len(subset)} samples remain")
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
        train_kd_loss = 0
        processed_batches = 0

        # ---- 分阶段解冻 ----
        setup_staged_freezing(
            raw_model, epoch, unfreeze_epoch=unfreeze_epoch,
            log_fn=logger.info if world_rank == 0 else lambda *a: None,
        )
        kd_alpha = get_kd_alpha(epoch, cfg.max_epoch, kd_alpha_start, kd_alpha_end)

        # 续训 epoch 用截断 DataLoader，后续 epoch 恢复全量 DataLoader
        if epoch == start_epoch and resume_dataloader is not None:
            current_dataloader = resume_dataloader
            current_offset = resume_batch_offset
        else:
            current_dataloader = original_train_dataloader
            current_offset = 0

        # train_max_batches / train_skip_stride 始终基于全量 DataLoader 计算
        train_max_batches = int(os.environ.get("TRAIN_MAX_BATCHES", "2940"))
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
                    loss1 = loss_func(out_surface, tar_surface, surface_weights, level_weight=surface_level_weight)
                    loss2 = loss_func(out_upper_air, tar_upper_air, pressure_weights, level_weight=1.0)

                    if kd_enabled and teacher_model is not None:
                        with torch.no_grad():
                            teacher_s, teacher_u = teacher_model(invar)
                            teacher_u = teacher_u.reshape(tar_upper_air.shape)
                        kd_loss = compute_kd_loss(
                            out_surface, out_upper_air,
                            teacher_s, teacher_u,
                            surface_weights, pressure_weights,
                            level_weight_surface=surface_level_weight,
                            level_weight_upper=1.0,
                        )
                    else:
                        kd_loss = torch.tensor(0.0, device=local_rank)
            # 总 loss = GT loss + α × KD loss（按累积步数缩放）
            total_gt_loss = loss1 + loss2
            total_loss = total_gt_loss + kd_alpha * kd_loss
            loss = total_loss / accumulation_steps
            loss.backward()
            train_loss += total_loss.item()
            train_kd_loss += kd_loss.item()
            processed_batches += 1
            if processed_batches % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch}-{already_processed + processed_batches}/{train_max_batches} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/processed_batches: .02f}s/{effective_batch_size}batch] '
                            f'loss:{train_loss / processed_batches: .04f}'
                            + (f' kd:{train_kd_loss / processed_batches: .04f}(α={kd_alpha:.2f})' if kd_enabled else ''))
                if save_interval_sec > 0 and time.time() - last_interval_save >= save_interval_sec:
                    save_checkpoint(
                        model, optimizer, scheduler, best_valid_loss, best_loss_epoch,
                        cfg.checkpoint_dir, latest_ckpt_name,
                        epoch=epoch, next_batch_idx=current_offset + j + 1, epoch_complete=False,
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
        val_max_samples = int(os.environ.get("VAL_MAX_SAMPLES", "500"))
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
                    loss1 = loss_func(out_surface, tar_surface, surface_weights, level_weight=surface_level_weight).item()
                    loss2 = loss_func(out_upper_air, tar_upper_air, pressure_weights, level_weight=1.0).item()
                # 总 loss
                loss = loss1 + loss2

                if cfg.world_size > 1:
                    loss_tensor = torch.tensor(loss, device=local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                valid_loss += loss
                val_processed += 1
                if world_rank == 0:
                    seen_samples = min(val_processed * cfg_data.dataloader.batch_size, val_max_samples)
                    logger.info(f'Valid: Epoch {epoch}-{val_processed}/{val_max_batches} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/val_processed: .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'samples:{seen_samples}/{val_max_samples} '
                            f'loss:{valid_loss / val_processed: .04f}')
                
        valid_loss /= max(1, val_processed)
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            if world_rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler, best_valid_loss, best_loss_epoch,
                    cfg.checkpoint_dir, best_ckpt_name,
                    epoch=epoch, next_batch_idx=0, epoch_complete=True,
                )
            is_save_ckp = True

        scheduler.step()

        if world_rank == 0:
            save_checkpoint(
                model, optimizer, scheduler, best_valid_loss, best_loss_epoch,
                cfg.checkpoint_dir, latest_ckpt_name,
                epoch=epoch, next_batch_idx=0, epoch_complete=True,
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


def save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path, filename, epoch=None, next_batch_idx=0, epoch_complete=True):
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
    }
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
