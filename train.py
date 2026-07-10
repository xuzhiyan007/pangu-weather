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

# 优先使用 apex FusedAdam（更快），不可用时回退到 torch.optim.AdamW
try:
    from apex import optimizers as _apex_optimizers
    FusedAdam = _apex_optimizers.FusedAdam
    print("✅ 使用 apex.FusedAdam 优化器")
except ImportError:
    FusedAdam = torch.optim.AdamW
    print("⚠️  apex 未安装，回退到 torch.optim.AdamW")


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

    surface_weights = torch.as_tensor(cfg_data.dataset.weights[:4], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)
    pressure_weights = torch.as_tensor(cfg_data.dataset.weights[4:], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)

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

    ## 剪枝检测
    is_pruned = (cfg.depth[1] < 6 or cfg.depth[2] < 6)
    ckpt_suffix = "_pruned" if is_pruned else ""

    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss_288{ckpt_suffix}.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss_288{ckpt_suffix}.npy"
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
        if is_pruned:
            pruned_l2 = 6 - cfg.depth[1]
            pruned_l3 = 6 - cfg.depth[2]
            print(f"✂️  Block 剪枝已启用: layer2={cfg.depth[1]}/6 (剪{pruned_l2}), layer3={cfg.depth[2]}/6 (剪{pruned_l3})")
        print("-" * 50, "\n")

    ## Checkpoint file names
    best_ckpt_name = f"model_288{ckpt_suffix}_bak.pth"
    latest_ckpt_name = f"model_288{ckpt_suffix}_latest.pth"
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, best_ckpt_name)
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, latest_ckpt_name)
    save_interval_sec = int(os.environ.get("SAVE_INTERVAL_SEC", "60"))

    ## Load model weight if there exist well-trained model
    start_epoch = 0
    start_batch = 0
    loaded_resume = False
    resume_ckpt_data = None
    freeze_transformer = False
    unfreeze_epoch = int(os.environ.get("UNFREEZE_EPOCH", "999"))

    # ---- Step 1: 断点续训（优先）：288 checkpoint = 继承全量模型训练后的结果 ----
    if os.path.exists(latest_ckpt_path) or os.path.exists(best_ckpt_path):
        resume_ckpt = latest_ckpt_path if os.path.exists(latest_ckpt_path) else best_ckpt_path
        if world_rank == 0:
            print("\n")
            print("-" * 50)
            print(f"loading resume checkpoint: {resume_ckpt}")
            print(f'删除此文件可重新从全量模型 warm start')
            print("-" * 50 + "\n")
        ckpt = torch.load(resume_ckpt, map_location=f'cuda:{local_rank}', weights_only=False)
        # 剪枝兼容：过滤掉 checkpoint 中不属于剪枝后模型的参数
        model_state = model.state_dict()
        src_state = ckpt["model_state_dict"]
        filtered_state = {k: v for k, v in src_state.items() if k in model_state and model_state[k].shape == v.shape}
        skipped_keys = [k for k in src_state if k not in model_state or (k in model_state and model_state[k].shape != src_state[k].shape)]
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        if world_rank == 0 and skipped_keys:
            print(f"✂️  续训跳过 {len(skipped_keys)} 个被剪枝参数: {skipped_keys[:5]}")
        resume_ckpt_data = ckpt
        best_valid_loss = ckpt.get("best_valid_loss", best_valid_loss)
        best_loss_epoch = ckpt.get("best_loss_epoch", best_loss_epoch)
        saved_epoch = int(ckpt.get("epoch", -1))
        if ckpt.get("epoch_complete", True):
            start_epoch = saved_epoch + 1
            start_batch = 0
        else:
            start_epoch = saved_epoch
            start_batch = int(ckpt.get("next_batch_idx", 0))
        if os.path.exists(train_loss_file):
            train_losses = np.load(train_loss_file)
        if os.path.exists(valid_loss_file):
            valid_losses = np.load(valid_loss_file)
        loaded_resume = True
        if world_rank == 0:
            print(f"resume from epoch {start_epoch}, batch {start_batch}")

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
        pruned_keys = []
        for k, v in src_state.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    shape_mismatch_keys.append(k)
            else:
                pruned_keys.append(k)

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        real_missing = [k for k in missing_keys if "attn_mask" not in k]
        attn_mask_missing = [k for k in missing_keys if "attn_mask" in k]
        if world_rank == 0:
            print(f"✅ 已迁移 {len(filtered_state)} 个参数（QKV/MLP/LayerNorm）")
            if shape_mismatch_keys:
                print(f"⚠️  {len(shape_mismatch_keys)} 个参数因 patch_size 变更跳过（形状不匹配）:")
                print(f"   {shape_mismatch_keys[:10]}")
            if pruned_keys:
                print(f"✂️  {len(pruned_keys)} 个参数因剪枝跳过（被移除的 block）:")
                print(f"   {pruned_keys[:10]}")
            print(f"   {len(real_missing)} 个参数随机初始化（patch_embed/recovery/downsample/upsample）")
            print(f"   attn_mask buffer {len(attn_mask_missing)} 个将自动重建")
            if real_missing:
                print(f"   missing keys (前20): {real_missing[:20]}")
            if unexpected_keys:
                print(f"   unexpected keys (前20): {list(unexpected_keys)[:20]}")
            print("   optimizer/scheduler 从头初始化，从 epoch 0 开始训练")
        start_epoch = 0
        start_batch = 0

    # ---- Step 2: 冻结策略（剪枝感知）----
    if is_pruned:
        # 剪枝模式：冻结 layer1，解冻 layer2/3/4（剪枝改变了结构，需重新训练）
        frozen_layers = ["layer1"]
        trainable_layers = ["layer2", "layer3", "layer4"]
    else:
        # 非剪枝模式：冻结所有 Transformer 层（原版行为）
        frozen_layers = ["layer1", "layer2", "layer3", "layer4"]
        trainable_layers = []

    freeze_transformer = True
    for name, param in model.named_parameters():
        if any(layer in name for layer in frozen_layers):
            param.requires_grad = False
        elif any(layer in name for layer in trainable_layers):
            param.requires_grad = True

    # 解冻 earth_position_bias_table（形状因 patch_size 变更，需要从头学习）
    for name, param in model.named_parameters():
        if "earth_position_bias_table" in name:
            param.requires_grad = True

    if world_rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if is_pruned:
            print(f"🔥 剪枝感知冻结：layer1 已冻结，layer2/3/4 已解冻微调，仅训练 {trainable}/{total} 参数 ({trainable/total*100:.1f}%)")
        else:
            print(f"🧊 Transformer 已冻结（QKV/MLP/LayerNorm），仅训练 {trainable}/{total} 参数 ({trainable/total*100:.1f}%)")
        print(f"   earth_position_bias_table 已单独解冻，将从 trunc_normal 初始化开始学习")
        if unfreeze_epoch < 999:
            print(f"   将在 epoch {unfreeze_epoch} 解冻全部参数")

    raw_model = model  # 保留原始模型引用，用于 replace_function
    if world_rank == 0:
        print("torch.compile disabled; training will use eager mode")

    ## Optimizer init（仅优化可训练参数，冻结的参数不进入 optimizer）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = FusedAdam(trainable_params, betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6)
    scheduler = build_scheduler(optimizer)

    ## 断点续训时恢复 optimizer/scheduler 状态
    if resume_ckpt_data is not None:
        optimizer.load_state_dict(resume_ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt_data["scheduler_state_dict"])
        if world_rank == 0:
            print("✅ optimizer/scheduler 状态已恢复")

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

        # 解冻逻辑：到达指定 epoch 时解除 Transformer 冻结，全量微调
        if freeze_transformer and epoch >= unfreeze_epoch:
            for param in raw_model.parameters():
                param.requires_grad = True
            optimizer = FusedAdam(
                [p for p in model.parameters() if p.requires_grad],
                betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6
            )
            scheduler = build_scheduler(optimizer, epoch=epoch)
            freeze_transformer = False
            if world_rank == 0:
                logger.info(f"🔓 epoch {epoch}: Transformer 已解冻，全量参数训练")

        model.train()
        train_loss = 0
        processed_batches = 0

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
                    loss1 = loss_func(out_surface, tar_surface, surface_weights,  level_weight=0.25)
                    loss2 = loss_func(out_upper_air, tar_upper_air, pressure_weights, level_weight=1.0)
            # 总 loss（按累积步数缩放，使梯度等价于大 batch）
            loss = (loss1 + loss2) / accumulation_steps
            loss.backward()
            train_loss += (loss.item() * accumulation_steps)  # 记录原始 loss 用于日志
            processed_batches += 1
            if processed_batches % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch}-{already_processed + processed_batches}/{train_max_batches} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/processed_batches: .02f}s/{effective_batch_size}batch] '
                            f'loss:{train_loss / processed_batches: .04f}')
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
        val_max_samples = int(os.environ.get("VAL_MAX_SAMPLES", "50"))
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
                    loss1 = loss_func(out_surface, tar_surface, surface_weights,  level_weight=0.25).item()
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
