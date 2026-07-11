"""
知识蒸馏工具模块。

提供以下功能：
  - init_patch_embed_from_official: 从官方 [2,4,4] 模型聚合初始化 [2,8,8] 的 patch_embed/recovery conv 权重
  - load_teacher_model:              加载官方模型作为 teacher（frozen, no_grad）
  - compute_kd_loss:                 输出层蒸馏 loss（weighted L1 between student & teacher）
  - setup_staged_freezing:           分阶段冻结/解冻 Transformer 主干
  - get_kd_alpha:                    KD loss 权重线性退火
"""

import torch
import torch.nn.functional as F
from pangu_patchsize import PanguPatchSize as Pangu


# ---------------------------------------------------------------------------
# 1. patch_embed / patch_recovery 智能初始化
# ---------------------------------------------------------------------------

def init_patch_embed_from_official(model, official_ckpt_path, device="cpu", log_fn=print):
    """
    从官方 [2,4,4] 模型聚合初始化 [2,8,8] 模型的 patch_embed / patch_recovery conv 权重。

    原理：
      - patch_embed (Conv3d/Conv2d): 输出 token = Σ(input × weight) over kernel。
        8×8 kernel 比 4×4 多 4 倍求和位置 → 复制后除以 4 保持输出幅度不变。
      - patch_recovery (ConvTranspose3d/ConvTranspose2d): 每个输出像素只从 1 个
        input token 获得 1 次贡献 → 复制空间维度即可，无需缩放。

    仅处理形状不匹配的 conv weight（形状匹配的已由 warm start 继承）。
    """
    ckpt = torch.load(official_ckpt_path, map_location=device, weights_only=False)
    src_state = ckpt.get("model_state_dict", ckpt)

    initialized = []
    for name, param in model.named_parameters():
        if "proj.weight" not in name or name not in src_state:
            continue

        src_w = src_state[name]
        if src_w.shape == param.shape:
            continue  # 形状相同，已由 warm start 继承
        if src_w.ndim < 4:
            continue  # 非 conv weight

        # 确定空间维度位置
        if src_w.ndim == 5:       # Conv3d [out, in, d, h, w]
            h_dim, w_dim = 3, 4
        elif src_w.ndim == 4:     # Conv2d [out, in, h, w]
            h_dim, w_dim = 2, 3
        else:
            continue

        src_h, src_w_val = src_w.shape[h_dim], src_w.shape[w_dim]
        tgt_h, tgt_w = param.shape[h_dim], param.shape[w_dim]
        if tgt_h != src_h * 2 or tgt_w != src_w_val * 2:
            continue  # 不是 2× 扩展，跳过

        # 将 4×4 权重复制到 8×8：每个位置展开为 2×2 子块
        new_w = src_w.repeat_interleave(2, dim=h_dim).repeat_interleave(2, dim=w_dim)

        is_recovery = "recovery" in name.lower()
        if not is_recovery:
            # Conv: sum over kernel → 4× 更多位置 → 除以 4 保持幅度
            new_w = new_w / 4.0
            # ConvTranspose: 每个输出像素只收 1 次贡献 → 无需缩放

        param.data.copy_(new_w)
        tag = "÷4" if not is_recovery else "no-scale"
        initialized.append(f"  ✅ {name}: {list(src_w.shape)}→{list(param.shape)} ({tag})")

    for line in initialized:
        log_fn(line)
    return len(initialized)


# ---------------------------------------------------------------------------
# 2. Teacher 模型加载
# ---------------------------------------------------------------------------

def load_teacher_model(ckpt_path, img_size, device,
                       depth=(2, 6, 6, 2), patch_size=(2, 4, 4),
                       embed_dim=192, num_heads=(6, 12, 12, 6),
                       window_size=(2, 6, 12), log_fn=print):
    """
    加载官方 [2,4,4] 模型作为 teacher。

    teacher 模型与 student 模型具有相同的 embed_dim / num_heads / window_size，
    唯一区别是 patch_size ([2,4,4] vs [2,8,8]) 和 depth（teacher 始终为完整 [2,6,6,2]）。

    teacher 输出形状与 student 完全一致：
      surface:    [B, 4, H, W]
      upper_air:  [B, 5, 13, H, W]  (reshape 后 [B, 65, H, W])
    """
    teacher = Pangu(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_size=window_size,
        depth=depth,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_state = ckpt.get("model_state_dict", ckpt)

    model_sd = teacher.state_dict()
    filtered = {k: v for k, v in src_state.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    missing, unexpected = teacher.load_state_dict(filtered, strict=False)
    bad_missing = [k for k in missing if "attn_mask" not in k]

    if bad_missing:
        log_fn(f"⚠️  Teacher missing critical keys: {bad_missing[:10]}")
    else:
        log_fn(f"✅ Teacher loaded: {len(filtered)} params, depth={depth}, patch_size={patch_size}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return teacher


# ---------------------------------------------------------------------------
# 3. KD loss 计算
# ---------------------------------------------------------------------------

def compute_kd_loss(student_surface, student_upper,
                    teacher_surface, teacher_upper,
                    surface_weights, pressure_weights,
                    level_weight_surface=1.0, level_weight_upper=1.0):
    """
    输出层蒸馏 loss: weighted L1(student, teacher)。

    使用与 GT loss 相同的 channel-level 权重（target15=1.0, non-target=clamped），
    确保 KD 梯度也聚焦在 15 个评测目标通道上。
    """
    kd_surface = level_weight_surface * (
        F.l1_loss(student_surface, teacher_surface.detach(), reduction="none") * surface_weights
    ).mean()

    kd_upper = level_weight_upper * (
        F.l1_loss(student_upper, teacher_upper.detach(), reduction="none") * pressure_weights
    ).mean()

    return kd_surface + kd_upper


# ---------------------------------------------------------------------------
# 4. 分阶段冻结
# ---------------------------------------------------------------------------

def setup_staged_freezing(model, epoch, unfreeze_epoch=2, log_fn=print):
    """
    Stage 1 (epoch < unfreeze_epoch): 冻结 Transformer 主干（layer1-4），
                                      只训练接口层（patch_embed/recovery/downsample/upsample/bias_table）。
    Stage 2 (epoch >= unfreeze_epoch): 解冻全部参数，全模型微调。

    原理：Transformer 权重已从官方完美继承（cos=1.0），先冻结可以让接口层
    快速对齐到 Transformer 的表征空间，避免接口层的随机梯度破坏 Transformer 权重。
    """
    if epoch < unfreeze_epoch:
        for name, param in model.named_parameters():
            is_backbone = any(layer in name for layer in ["layer1", "layer2", "layer3", "layer4"])
            param.requires_grad = not is_backbone
        if epoch == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            log_fn(f"🔒 Stage 1 (epoch {epoch}<{unfreeze_epoch}): backbone frozen, "
                   f"trainable {trainable}/{total} ({trainable / total * 100:.1f}%)")
    else:
        for param in model.parameters():
            param.requires_grad = True
        if epoch == unfreeze_epoch:
            log_fn(f"🔓 Stage 2 (epoch {epoch}>={unfreeze_epoch}): full fine-tuning")


# ---------------------------------------------------------------------------
# 5. KD alpha 退火
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. 接口层重置（不删除 288 checkpoint，参照官方模型重置接口层）
# ---------------------------------------------------------------------------

def reset_interface_from_official(model, official_ckpt_path, device="cpu", log_fn=print):
    """
    从官方 [2,4,4] 模型重置 [2,8,8] 模型的接口层参数。

    适用场景：已有训练过的 288 checkpoint，不想删除重训，而是把接口层
    重置到接近官方的状态，再仅训练接口层进行微调对齐。

    重置策略：
      layer                  | 形状   | 操作                 | 原因
      -----------------------|--------|----------------------|----------------
      Transformer (layer1-4) | 匹配   | 不动                 | 已完美继承 cos=1.0
      downsample (linear/norm)| 匹配  | 直接复制官方         | 修复训练漂移 cos=0.83
      upsample (linear1/2/norm)|匹配  | 直接复制官方         | 修复训练漂移 cos=0.84
      patch_embed proj.bias  | 匹配   | 直接复制官方         | 修复漂移 cos=0.89
      patch_recovery proj.bias| 匹配  | 直接复制官方         | 修复漂移 cos=0.92
      patch_embed proj.weight| 不匹配 | 4×4→8×8 聚合 ÷4     | 结构性差异
      patch_recovery proj.weight|不匹配| 4×4→8×8 聚合不缩放  | 结构性差异
      earth_position_bias_table| 不匹配| 保留 288 checkpoint | 已训练且无法复制
    """
    ckpt = torch.load(official_ckpt_path, map_location=device, weights_only=False)
    src_state = ckpt.get("model_state_dict", ckpt)

    copied = []
    aggregated = []
    skipped = []

    for name, param in model.named_parameters():
        # 跳过 Transformer 主干（已完美继承，不动）
        if any(layer in name for layer in ["layer1", "layer2", "layer3", "layer4"]):
            continue

        # earth_position_bias_table：形状不匹配，保留 288 checkpoint 值
        if "position_bias" in name.lower():
            skipped.append(f"  ⏭️  {name}: kept from 288 checkpoint")
            continue

        if name not in src_state:
            continue

        src_w = src_state[name]

        if src_w.shape == param.shape:
            # 形状匹配：直接复制官方权重（修复 downsample/upsample/bias 漂移）
            param.data.copy_(src_w)
            copied.append(f"  📋 {name}: copied from official")

        elif "proj.weight" in name and src_w.ndim >= 4:
            # Conv weight 形状不匹配：4×4 → 8×8 聚合初始化
            if src_w.ndim == 5:
                h_dim, w_dim = 3, 4
            elif src_w.ndim == 4:
                h_dim, w_dim = 2, 3
            else:
                continue

            src_h, src_w_val = src_w.shape[h_dim], src_w.shape[w_dim]
            tgt_h, tgt_w = param.shape[h_dim], param.shape[w_dim]
            if tgt_h != src_h * 2 or tgt_w != src_w_val * 2:
                continue

            new_w = src_w.repeat_interleave(2, dim=h_dim).repeat_interleave(2, dim=w_dim)
            is_recovery = "recovery" in name.lower()
            if not is_recovery:
                new_w = new_w / 4.0

            param.data.copy_(new_w)
            tag = "÷4" if not is_recovery else "no-scale"
            aggregated.append(f"  🔧 {name}: {list(src_w.shape)}→{list(param.shape)} ({tag})")

    log_fn(f"✅ Interface reset complete:")
    log_fn(f"   {len(copied)} params copied from official (downsample/upsample/bias)")
    log_fn(f"   {len(aggregated)} conv weights aggregated (patch_embed/recovery)")
    log_fn(f"   {len(skipped)} params kept from 288 checkpoint (bias_table)")
    for line in copied + aggregated + skipped:
        log_fn(line)

    return len(copied) + len(aggregated)


def get_kd_alpha(epoch, max_epoch, alpha_start=1.0, alpha_end=0.3):
    """
    KD loss 权重线性退火：从 alpha_start 线性衰减到 alpha_end。

    早期训练时 KD 权重高 → 快速对齐 teacher；
    后期 KD 权重降低 → 让 GT loss 主导，使模型也有机会从真实标签学习。
    """
    if max_epoch <= 1:
        return alpha_start
    return alpha_start + (alpha_end - alpha_start) * epoch / (max_epoch - 1)
