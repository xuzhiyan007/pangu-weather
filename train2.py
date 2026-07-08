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
from apex import optimizers
from torch.utils.data import Subset
from torch.cuda.amp import autocast


def loss_func(x, y, weights, level_weight=1.0):
    return level_weight * (F.l1_loss(x, y, reduction='none') * weights).mean()

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
    config_file_path = os.path.join(current_path, "conf/config2.yaml")
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
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)

    ## Model init
    model = Pangu(img_size=cfg_data.dataset.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  num_heads=cfg.num_heads,
                  window_size=cfg.window_size,
                  ).to(local_rank)
    
    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss_266.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss_266.npy"
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
    best_ckpt_name = "model_266_bak.pth"
    latest_ckpt_name = "model_266_latest.pth"
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, best_ckpt_name)
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, latest_ckpt_name)
    save_interval_sec = int(os.environ.get("SAVE_INTERVAL_SEC", "600"))

    ## Load model weight if there exist well-trained model
    start_epoch = 0
    start_batch = 0
    loaded_resume = False
    resume_ckpt_data = None
    freeze_transformer = False
    unfreeze_epoch = int(os.environ.get("UNFREEZE_EPOCH", "999"))

    # ---- Step 1: 断点续训（优先）：266 checkpoint = 继承全量模型训练后的结果 ----
    if os.path.exists(latest_ckpt_path) or os.path.exists(best_ckpt_path):
        resume_ckpt = latest_ckpt_path if os.path.exists(latest_ckpt_path) else best_ckpt_path
        if world_rank == 0:
            print("\n")
            print("-" * 50)
            print(f"loading resume checkpoint: {resume_ckpt}")
            print(f'删除此文件可重新从全量模型 warm start')
            print("-" * 50 + "\n")
        ckpt = torch.load(resume_ckpt, map_location=f'cuda:{local_rank}', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
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

    # ---- Step 1 (备): warm start：无 266 checkpoint 时，从 244 全量权重部分迁移 ----
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

    # ---- Step 2: 无论哪种加载方式，都冻结 Transformer 层 ----
    freeze_transformer = True
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["layer1", "layer2", "layer3", "layer4"]):
            param.requires_grad = False

    # 解冻 earth_position_bias_table（形状因 patch_size 变更，需要从头学习）
    for name, param in model.named_parameters():
        if "earth_position_bias_table" in name:
            param.requires_grad = True

    if world_rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"🧊 Transformer 已冻结（QKV/MLP/LayerNorm），仅训练 {trainable}/{total} 参数 ({trainable/total*100:.1f}%)")
        print(f"   earth_position_bias_table 已单独解冻，将从 trunc_normal 初始化开始学习")
        if unfreeze_epoch < 999:
            print(f"   将在 epoch {unfreeze_epoch} 解冻全部参数")

    ## torch.compile (PyTorch 2.0+): 算子融合 + 内核优化
    raw_model = model  # 保留原始模型引用，用于 replace_function
    use_compile = os.environ.get("DISABLE_COMPILE", "0") != "1"
    if use_compile:
        import importlib
        _dynamo = importlib.import_module("torch._dynamo")
        _dynamo.config.suppress_errors = True  # 编译失败时自动回退 eager 模式
        model = torch.compile(model)
        if world_rank == 0:
            print("⚡ torch.compile 已启用（编译失败时自动回退 eager）")
    else:
        if world_rank == 0:
            print("⚠️ torch.compile 已禁用（DISABLE_COMPILE=1）")

    ## Optimizer init（仅优化可训练参数，冻结的参数不进入 optimizer）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizers.FusedAdam(trainable_params, betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    ## 断点续训时恢复 optimizer/scheduler 状态
    if resume_ckpt_data is not None:
        optimizer.load_state_dict(resume_ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt_data["scheduler_state_dict"])
        if world_rank == 0:
            print("✅ optimizer/scheduler 状态已恢复")

    # 续训时重建 DataLoader 跳过已处理 batch，避免逐个加载数据
    if start_batch > 0 and train_sampler is None:
        dataset = train_dataloader.dataset
        skip_samples = start_batch * cfg_data.dataloader.batch_size
        subset = Subset(dataset, range(skip_samples, len(dataset)))
        train_dataloader = torch.utils.data.DataLoader(
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
        if freeze_transformer and epoch == unfreeze_epoch:
            for param in raw_model.parameters():
                param.requires_grad = True
            optimizer = optimizers.FusedAdam(
                [p for p in model.parameters() if p.requires_grad],
                betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6
            )
            freeze_transformer = False
            if world_rank == 0:
                logger.info(f"🔓 epoch {epoch}: Transformer 已解冻，全量参数训练")

        model.train()
        train_loss = 0
        processed_batches = 0
        train_max_batches = int(os.environ.get("TRAIN_MAX_BATCHES", "2940"))
        train_total = len(train_dataloader)
        if train_max_batches > 0 and train_max_batches < train_total:
            train_skip_stride = max(2, round(train_total / (train_total - train_max_batches)))
        else:
            train_skip_stride = 0
            train_max_batches = train_total
        start_time = time.time()
        last_interval_save = time.time()
        if epoch == start_epoch and start_batch > 0:
            logger.info(f"skipping first {start_batch} batches to resume...")
        for j, data in enumerate(train_dataloader):
            if epoch == start_epoch and j < start_batch:
                logger.info(f"  skip progress: {j}/{start_batch} batches")
                continue
            if train_skip_stride > 0 and j % train_skip_stride == (epoch % train_skip_stride):
                continue
            invar = data[0]
            outvar = data[1]
            invar_surface = invar[:, :4, :, :].to(local_rank, dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to(local_rank, dtype=torch.float32)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)
            tar_surface = outvar[:, :4, :, :].to(local_rank, dtype=torch.float32)
            tar_upper_air = outvar[:, 4:, :, :].to(local_rank, dtype=torch.float32)

            with replace_function(raw_model,["layer2", "layer3"],cfg.world_size > 1):
                with autocast(dtype=torch.bfloat16):
                    out_surface, out_upper_air = model(invar)
                    out_upper_air = out_upper_air.reshape(tar_upper_air.shape)
                    loss1 = loss_func(out_surface, tar_surface, surface_weights,  level_weight=0.25)
                    loss2 = loss_func(out_upper_air, tar_upper_air, pressure_weights, level_weight=1.0)
            # 总 loss
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            processed_batches += 1
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch}-{processed_batches}/{train_max_batches} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/processed_batches: .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{train_loss / processed_batches: .04f}')
                if save_interval_sec > 0 and time.time() - last_interval_save >= save_interval_sec:
                    save_checkpoint(
                        model, optimizer, scheduler, best_valid_loss, best_loss_epoch,
                        cfg.checkpoint_dir, latest_ckpt_name,
                        epoch=epoch, next_batch_idx=j + 1, epoch_complete=False,
                    )
                    logger.info(f"interval checkpoint saved at epoch={epoch}, next_batch={j + 1}")
                    last_interval_save = time.time()
            
        train_loss = train_loss / max(1, processed_batches)

        model.eval()
        valid_loss = 0
        val_max_batches = int(os.environ.get("VAL_MAX_BATCHES", "400"))
        val_total = len(val_dataloader)
        if val_max_batches > 0 and val_max_batches < val_total:
            val_stride = max(1, val_total // val_max_batches)
        else:
            val_stride = 1
            val_max_batches = val_total
        val_processed = 0
        with torch.no_grad():
            start_time = time.time()
            for j, data in enumerate(val_dataloader):
                if val_stride > 1 and j % val_stride != 0:
                    continue
                invar = data[0]
                outvar = data[1]
                invar_surface = invar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                invar_upper_air = invar[:, 4:, :, :].to(local_rank, dtype=torch.float32)
                invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)
                tar_surface = outvar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                tar_upper_air = outvar[:, 4:, :, :].to(local_rank, dtype=torch.float32)

                with replace_function(raw_model,["layer2", "layer3"],cfg.world_size > 1):
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
                    logger.info(f'Valid: Epoch {epoch}-{val_processed}/{val_max_batches} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/val_processed: .02f}s/{cfg_data.dataloader.batch_size}batch] '
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
