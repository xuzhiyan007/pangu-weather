"""
通道剪枝权重迁移脚本

从 channel_importance.json 读取通道重要性分数，
从头对齐方式选择保留的 head，按通道索引从源 checkpoint
抽取权重到通道剪枝后的模型。

头对齐剪枝：保持 head_dim=32 不变，按 head 重要性选择保留的 head，
确保 QKV/Proj/PositionBias 的结构一致性。

用法:
    python channel_transfer.py \
        --ckpt data/checkpoints/model_288_pruned2642_latest.pth \
        --importance analysis/channel_importance.json \
        --target-dim 288 \
        --target-heads 9 \
        --output data/checkpoints/model_288_channel288_init.pth

    然后微调:
        CHANNEL_PRUNE_FROM=data/checkpoints/model_288_channel288_init.pth python train.py
"""

import argparse
import json
import os
import sys

import torch
import numpy as np
from torch import nn

# 确保能导入 pangu_patchsize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pangu_patchsize import PanguPatchSize as Pangu
from onescience.utils.YParams import YParams


# ──────────────────────────────────────────────
# 1. 头对齐通道选择
# ──────────────────────────────────────────────

def compute_head_aligned_keep(emb_scores_l2, emb_scores_l3,
                               orig_dim, orig_heads, target_heads):
    """
    合并 layer2/layer3 的通道重要性，按 head 对齐方式选择保留的 head。

    因为 layer2 → layer3 顺序连接，两层必须使用相同的通道索引。
    所以将两层的 per-head 重要性取平均后选择 top-k head。

    Args:
        emb_scores_l2: layer2 的 embedding 重要性分数 [orig_dim]
        emb_scores_l3: layer3 的 embedding 重要性分数 [orig_dim]
        orig_dim: 原始维度 (384)
        orig_heads: 原始头数 (12)
        target_heads: 目标头数 (9)

    Returns:
        keep_channels: list[int], 保留的通道索引（排序后），长度 = target_heads * head_dim
        keep_heads: list[int], 保留的 head 索引（排序后）
    """
    head_dim = orig_dim // orig_heads  # 32
    s_l2 = torch.tensor(emb_scores_l2, dtype=torch.float32)
    s_l3 = torch.tensor(emb_scores_l3, dtype=torch.float32)

    # per-head 重要性 = head 内通道重要性之和
    head_imp_l2 = s_l2.reshape(orig_heads, head_dim).sum(dim=1)  # [orig_heads]
    head_imp_l3 = s_l3.reshape(orig_heads, head_dim).sum(dim=1)
    # 两层取平均
    head_importance = (head_imp_l2 + head_imp_l3) / 2.0

    # 保留最重要的 target_heads 个 head
    keep_heads = head_importance.topk(target_heads).indices.sort().values.tolist()

    # 展开为通道索引
    keep_channels = []
    for h in keep_heads:
        keep_channels.extend(range(h * head_dim, (h + 1) * head_dim))

    # 打印每个 head 的重要性
    print("\n头对齐通道选择:")
    print(f"  head_dim={head_dim}, orig_heads={orig_heads}, target_heads={target_heads}")
    print(f"  per-head importance (layer2+layer3 avg):")
    for h in range(orig_heads):
        marker = "✓" if h in keep_heads else "✗"
        print(f"    head {h:2d}: {head_importance[h].item():.4f} {marker}")
    print(f"  保留 head: {keep_heads}")
    print(f"  保留通道数: {len(keep_channels)}")

    return keep_channels, keep_heads


def compute_mlp_keep(mlp_scores_l2, mlp_scores_l3,
                     orig_hidden, target_hidden):
    """
    合并 layer2/layer3 的 MLP 隐藏层重要性，选择保留的通道。

    MLP hidden 通道是每个 block 内部的，理论上各 block 可以不同。
    但为简化实现，layer2/layer3 所有 block 使用相同的 MLP keep indices。

    Args:
        mlp_scores_l2: layer2 的 MLP hidden 重要性 [orig_hidden]
        mlp_scores_l3: layer3 的 MLP hidden 重要性 [orig_hidden]
        orig_hidden: 原始 MLP hidden 维度 (1536 = 4*384)
        target_hidden: 目标 MLP hidden 维度 (1152 = 4*288)

    Returns:
        keep_indices: list[int], 保留的 MLP hidden 通道索引（排序后）
    """
    s_l2 = torch.tensor(mlp_scores_l2, dtype=torch.float32)
    s_l3 = torch.tensor(mlp_scores_l3, dtype=torch.float32)
    # 两层取平均
    combined = (s_l2 + s_l3) / 2.0

    keep_indices = combined.topk(target_hidden).indices.sort().values.tolist()
    print(f"\nMLP hidden 剪枝: {orig_hidden} → {target_hidden}")
    return keep_indices


# ──────────────────────────────────────────────
# 2. 权重切片函数
# ──────────────────────────────────────────────

def slice_qkv_weight(weight, keep_channels, keep_heads, orig_heads, head_dim):
    """
    QKV 权重切片。
    weight: [3*orig_heads*head_dim, orig_dim] = [1152, 384]
    → [3*target_heads*head_dim, target_dim] = [864, 288]
    """
    w = weight.reshape(3, orig_heads, head_dim, -1)
    # 切 head 维 (dim=1)
    w = w[:, keep_heads, :, :]
    # 切输入通道维 (dim=3)
    w = w[:, :, :, keep_channels]
    target_heads = len(keep_heads)
    target_dim = len(keep_channels)
    return w.reshape(3 * target_heads * head_dim, target_dim)


def slice_qkv_bias(bias, keep_heads, orig_heads, head_dim):
    """
    QKV bias 切片。
    bias: [3*orig_heads*head_dim] = [1152]
    → [3*target_heads*head_dim] = [864]
    """
    b = bias.reshape(3, orig_heads, head_dim)
    b = b[:, keep_heads, :]
    target_heads = len(keep_heads)
    return b.reshape(3 * target_heads * head_dim)


def slice_proj_weight(weight, keep_channels):
    """
    Proj 权重切片。
    weight: [orig_dim, orig_dim] = [384, 384]
    → [target_dim, target_dim] = [288, 288]

    输入侧 (cols) 是注意力输出 [num_heads*head_dim]，按 keep_channels 切。
    输出侧 (rows) 是 dim，按 keep_channels 切。
    因为 keep_channels 是头对齐的，所以行列切法一致。
    """
    return weight[torch.tensor(keep_channels)][:, torch.tensor(keep_channels)]


def slice_norm(weight_or_bias, keep_channels):
    """LayerNorm weight/bias 切片。 [orig_dim] → [target_dim]"""
    return weight_or_bias[torch.tensor(keep_channels)]


def slice_mlp_fc1_weight(weight, mlp_keep_indices, keep_channels):
    """
    MLP fc1 权重切片。
    weight: [4*orig_dim, orig_dim] = [1536, 384]
    → [4*target_dim, target_dim] = [1152, 288]
    rows = MLP hidden (按 mlp_keep_indices 切)
    cols = input dim (按 keep_channels 切)
    """
    return weight[torch.tensor(mlp_keep_indices)][:, torch.tensor(keep_channels)]


def slice_mlp_fc1_bias(bias, mlp_keep_indices):
    """MLP fc1 bias 切片。 [4*orig_dim] → [4*target_dim]"""
    return bias[torch.tensor(mlp_keep_indices)]


def slice_mlp_fc2_weight(weight, keep_channels, mlp_keep_indices):
    """
    MLP fc2 权重切片。
    weight: [orig_dim, 4*orig_dim] = [384, 1536]
    → [target_dim, 4*target_dim] = [288, 1152]
    rows = output dim (按 keep_channels 切)
    cols = MLP hidden (按 mlp_keep_indices 切)
    """
    return weight[torch.tensor(keep_channels)][:, torch.tensor(mlp_keep_indices)]


def slice_mlp_fc2_bias(bias, keep_channels):
    """MLP fc2 bias 切片。 [orig_dim] → [target_dim]"""
    return bias[torch.tensor(keep_channels)]


def slice_position_bias(table, keep_heads):
    """
    Position bias table 切片。
    table: [pos_size, nW, orig_heads]
    → [pos_size, nW, target_heads]
    """
    return table[:, :, keep_heads]


def slice_downsample_linear(weight, keep_channels):
    """
    Downsample linear 权重切片。
    weight: [orig_dim, 4*in_dim] = [384, 768]
    → [target_dim, 4*in_dim] = [288, 768]
    rows = output dim (按 keep_channels 切)
    cols = input (4*in_dim=768, 不变)
    """
    return weight[torch.tensor(keep_channels)]


def slice_upsample_linear1(weight, keep_channels):
    """
    Upsample linear1 权重切片。
    weight: [4*out_dim, orig_dim] = [768, 384]
    → [4*out_dim, target_dim] = [768, 288]
    rows = output (4*out_dim=768, 不变)
    cols = input dim (按 keep_channels 切)
    """
    return weight[:, torch.tensor(keep_channels)]


# ──────────────────────────────────────────────
# 3. 键名分类与转移
# ──────────────────────────────────────────────

def normalize_key(key):
    """将 .Fuser. 替换为 .fuser.，.Sampler. 替换为 .sampler.，用于分类"""
    return key.replace(".Fuser.", ".fuser.").replace(".Sampler.", ".sampler.")


def classify_param_key(key):
    """
    分类参数键，返回 (layer, wtype) 或 None。
    """
    nk = normalize_key(key)

    # 判断 layer
    layer = None
    if "layer1." in nk:
        layer = "layer1"
    elif "layer2." in nk:
        layer = "layer2"
    elif "layer3." in nk:
        layer = "layer3"
    elif "layer4." in nk:
        layer = "layer4"
    elif "downsample." in nk:
        layer = "downsample"
    elif "upsample." in nk:
        layer = "upsample"

    # 判断 wtype
    wtype = None
    if "earth_position_bias_table" in nk:
        wtype = "pos_bias"
    elif "qkv.weight" in nk:
        wtype = "qkv_weight"
    elif "qkv.bias" in nk:
        wtype = "qkv_bias"
    elif "attn" in nk and "proj.weight" in nk:
        wtype = "proj_weight"
    elif "attn" in nk and "proj.bias" in nk:
        wtype = "proj_bias"
    elif "mlp" in nk and "fc1.weight" in nk:
        wtype = "fc1_weight"
    elif "mlp" in nk and "fc1.bias" in nk:
        wtype = "fc1_bias"
    elif "mlp" in nk and "fc2.weight" in nk:
        wtype = "fc2_weight"
    elif "mlp" in nk and "fc2.bias" in nk:
        wtype = "fc2_bias"
    elif "norm1.weight" in nk or "norm1.bias" in nk:
        wtype = "norm1"
    elif "norm2.weight" in nk or "norm2.bias" in nk:
        wtype = "norm2"
    elif layer == "downsample" and "linear.weight" in nk:
        wtype = "downsample_linear"
    elif layer == "downsample" and ("norm.weight" in nk or "norm.bias" in nk):
        wtype = "downsample_norm"
    elif layer == "upsample" and "linear1.weight" in nk:
        wtype = "upsample_linear1"
    elif layer == "upsample" and "linear2.weight" in nk:
        wtype = "upsample_linear2"
    elif layer == "upsample" and ("norm.weight" in nk or "norm.bias" in nk):
        wtype = "upsample_norm"

    return layer, wtype


def transfer_param(key, src_tensor, keep_channels, keep_heads, mlp_keep_indices,
                   orig_heads, head_dim):
    """
    根据参数类型对源张量进行切片，返回转移后的张量。
    如果参数不需要切片（layer1/4 或非剪枝层），返回 None。
    """
    layer, wtype = classify_param_key(key)

    # layer1/4 和非剪枝层参数不需要切片
    if layer in ("layer1", "layer4"):
        return src_tensor
    if layer is None:
        return src_tensor

    # layer2/3 参数需要切片
    if layer in ("layer2", "layer3"):
        if wtype == "qkv_weight":
            return slice_qkv_weight(src_tensor, keep_channels, keep_heads,
                                    orig_heads, head_dim)
        elif wtype == "qkv_bias":
            return slice_qkv_bias(src_tensor, keep_heads, orig_heads, head_dim)
        elif wtype == "proj_weight":
            return slice_proj_weight(src_tensor, keep_channels)
        elif wtype == "proj_bias":
            return slice_norm(src_tensor, keep_channels)
        elif wtype in ("norm1", "norm2"):
            return slice_norm(src_tensor, keep_channels)
        elif wtype == "fc1_weight":
            return slice_mlp_fc1_weight(src_tensor, mlp_keep_indices, keep_channels)
        elif wtype == "fc1_bias":
            return slice_mlp_fc1_bias(src_tensor, mlp_keep_indices)
        elif wtype == "fc2_weight":
            return slice_mlp_fc2_weight(src_tensor, keep_channels, mlp_keep_indices)
        elif wtype == "fc2_bias":
            return slice_mlp_fc2_bias(src_tensor, keep_channels)
        elif wtype == "pos_bias":
            return slice_position_bias(src_tensor, keep_heads)
        else:
            # 未知类型，原样返回
            return src_tensor

    # downsample 参数
    if layer == "downsample":
        if wtype == "downsample_linear":
            return slice_downsample_linear(src_tensor, keep_channels)
        else:
            # norm 等不变
            return src_tensor

    # upsample 参数
    if layer == "upsample":
        if wtype == "upsample_linear1":
            return slice_upsample_linear1(src_tensor, keep_channels)
        else:
            # linear2, norm 等不变
            return src_tensor

    return src_tensor


# ──────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="通道剪枝权重迁移")
    parser.add_argument("--ckpt", required=True, help="源 checkpoint 路径")
    parser.add_argument("--importance", required=True,
                        help="channel_importance.json 路径")
    parser.add_argument("--target-dim", type=int, default=288,
                        help="目标通道维度 (默认 288)")
    parser.add_argument("--target-heads", type=int, default=9,
                        help="目标注意力头数 (默认 9)")
    parser.add_argument("--output", default=None,
                        help="输出 checkpoint 路径 (默认自动生成)")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Source checkpoint not found: {args.ckpt}")
    if not os.path.exists(args.importance):
        raise FileNotFoundError(f"Importance file not found: {args.importance}")

    # 读取 config
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, "conf", "config.yaml")
    cfg = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")

    embed_dim = cfg.embed_dim  # 192
    orig_dim = embed_dim * 2   # 384
    orig_heads = cfg.num_heads[1]  # 12
    head_dim = orig_dim // orig_heads  # 32
    target_dim = args.target_dim  # 288
    target_heads = args.target_heads  # 9
    target_hidden = target_dim * 4  # 1152
    orig_hidden = orig_dim * 4  # 1536

    print("=" * 60)
    print("  通道剪枝权重迁移")
    print("=" * 60)
    print(f"  源 checkpoint:   {args.ckpt}")
    print(f"  重要性文件:      {args.importance}")
    print(f"  原始 dim/heads:  {orig_dim} / {orig_heads} (head_dim={head_dim})")
    print(f"  目标 dim/heads:  {target_dim} / {target_heads} (head_dim={head_dim})")
    print(f"  MLP hidden:      {orig_hidden} → {target_hidden}")

    # 1. 读取通道重要性
    print("\n📂 读取通道重要性...")
    with open(args.importance, "r", encoding="utf-8") as f:
        importance = json.load(f)

    layers = importance["layers"]
    emb_scores_l2 = layers["layer2"]["embedding_importance"]["scores"]
    emb_scores_l3 = layers["layer3"]["embedding_importance"]["scores"]

    # MLP hidden 重要性
    mlp_scores_l2 = None
    mlp_scores_l3 = None
    if "mlp_hidden" in layers["layer2"]:
        # 从 recommendations 中获取 importance scores
        # channel_importance.json 存储的是 keep_indices，不是 scores
        # 我们需要从 raw scores 获取
        pass

    # 实际上 channel_importance.json 没有直接存储 MLP hidden scores
    # 但 recommendations 里有 keep_indices，我们可以用这些
    # 或者从源 checkpoint 重新计算
    # 这里我们选择从源 checkpoint 重新计算 MLP hidden 重要性

    # 2. 计算头对齐的通道保留索引
    keep_channels, keep_heads = compute_head_aligned_keep(
        emb_scores_l2, emb_scores_l3, orig_dim, orig_heads, target_heads
    )

    # 3. 从源 checkpoint 计算 MLP hidden 重要性
    print("\n📂 加载源 checkpoint 计算 MLP 重要性...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    src_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # 计算 MLP hidden 重要性（L1-norm 聚合）
    mlp_scores = torch.zeros(orig_hidden, dtype=torch.float32)
    mlp_count = 0
    for key, tensor in src_state.items():
        if not torch.is_tensor(tensor):
            continue
        nk = normalize_key(key)
        if ".fuser." not in nk and ".Fuser." not in key:
            continue
        if "mlp" not in nk:
            continue
        if "fc1.weight" in nk:
            # [4*dim, dim] → 行 L1 norm = MLP hidden 重要性
            imp = tensor.detach().float().abs().sum(dim=1)
            if imp.numel() == orig_hidden:
                mlp_scores += imp
                mlp_count += 1
        elif "fc2.weight" in nk:
            # [dim, 4*dim] → 列 L1 norm = MLP hidden 重要性
            imp = tensor.detach().float().abs().sum(dim=0)
            if imp.numel() == orig_hidden:
                mlp_scores += imp
                mlp_count += 1

    if mlp_count > 0:
        mlp_scores /= mlp_count
        mlp_keep_indices = compute_mlp_keep(
            mlp_scores, mlp_scores, orig_hidden, target_hidden
        )
    else:
        # 如果没有找到 MLP 权重，均匀选择
        print("⚠️  未找到 MLP 权重，使用均匀选择")
        mlp_keep_indices = list(range(target_hidden))

    # 4. 构建通道剪枝模型
    print("\n🔨 构建通道剪枝模型...")
    model = Pangu(
        img_size=cfg_data.dataset.img_size,
        patch_size=cfg.patch_size,
        embed_dim=embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        depth=cfg.depth,
        pruned_dim=target_dim,
        pruned_num_heads=target_heads,
    )

    model_state = model.state_dict()
    print(f"   目标模型参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   目标 state_dict 键数: {len(model_state)}")

    # 5. 转移权重
    print("\n🔄 转移权重...")
    transferred = {}
    skipped = []
    shape_mismatch = []

    for key, src_tensor in src_state.items():
        if not torch.is_tensor(src_tensor):
            continue

        # 跳过 buffer（attn_mask, earth_position_index 等）
        # 这些会在模型初始化时自动创建
        if "attn_mask" in key or "earth_position_index" in key:
            continue

        # 切片
        sliced = transfer_param(
            key, src_tensor, keep_channels, keep_heads, mlp_keep_indices,
            orig_heads, head_dim
        )

        if key in model_state:
            if sliced.shape == model_state[key].shape:
                transferred[key] = sliced
            else:
                shape_mismatch.append((key, tuple(sliced.shape), tuple(model_state[key].shape)))
        else:
            skipped.append(key)

    print(f"   ✅ 转移 {len(transferred)} 个参数")
    if shape_mismatch:
        print(f"   ⚠️  {len(shape_mismatch)} 个参数形状不匹配:")
        for key, src_shape, dst_shape in shape_mismatch[:10]:
            print(f"      {key}: {src_shape} → {dst_shape}")
    if skipped:
        print(f"   ℹ️  {len(skipped)} 个源参数不在目标模型中 (跳过)")

    # 6. 加载到模型
    print("\n📥 加载转移后的权重到模型...")
    missing_keys, unexpected_keys = model.load_state_dict(transferred, strict=False)
    real_missing = [k for k in missing_keys if "attn_mask" not in k and "earth_position_index" not in k]
    buffer_missing = [k for k in missing_keys if "attn_mask" in k or "earth_position_index" in k]

    if real_missing:
        print(f"   ⚠️  {len(real_missing)} 个参数未转移（随机初始化）:")
        for k in real_missing[:10]:
            print(f"      {k}")
    if unexpected_keys:
        print(f"   ⚠️  {len(unexpected_keys)} 个意外参数:")
        for k in unexpected_keys[:10]:
            print(f"      {k}")
    print(f"   buffer 自动重建: {len(buffer_missing)} 个 (attn_mask/position_index)")

    # 7. 保存 checkpoint
    output_path = args.output
    if output_path is None:
        ckpt_dir = cfg.checkpoint_dir
        output_path = os.path.join(ckpt_dir, f"model_288_channel{target_dim}_init.pth")

    # 保存为 train.py 可用的格式
    save_state = {
        "model_state_dict": model.state_dict(),
        "epoch": -1,
        "loss_version": "channel_pruned_init",
    }

    tmp_path = output_path + ".tmp"
    torch.save(save_state, tmp_path)
    os.replace(tmp_path, output_path)

    print(f"\n✅ 通道剪枝 checkpoint 已保存到: {output_path}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M)")

    # 保存剪枝元信息
    meta_path = output_path.replace(".pth", "_meta.json")
    meta = {
        "source_checkpoint": args.ckpt,
        "orig_dim": orig_dim,
        "orig_heads": orig_heads,
        "target_dim": target_dim,
        "target_heads": target_heads,
        "head_dim": head_dim,
        "keep_heads": keep_heads,
        "keep_channels_count": len(keep_channels),
        "mlp_keep_count": len(mlp_keep_indices),
        "params": sum(p.numel() for p in model.parameters()),
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   元信息已保存到: {meta_path}")

    print(f"\n📋 下一步:")
    print(f"   CHANNEL_PRUNE_FROM={output_path} python train.py")


if __name__ == "__main__":
    main()
