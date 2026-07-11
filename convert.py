import argparse
import os

import torch


POSITION_BIAS_NAME = "earth_position_bias_table"
ATTN_MASK_NAME = "attn_mask"
MLP_WEIGHT_PATTERNS = (
    ".transformer.mlp.fc1.weight",
    ".transformer.mlp.fc2.weight",
)


def quantize_symmetric_int8_per_tensor(tensor):
    t = tensor.detach().cpu().float()
    abs_max = t.abs().max().clamp(min=1e-8)
    scale = (abs_max / 127.0).to(torch.float16)
    q = torch.round(t / scale.float()).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale


def quantize_symmetric_int8_per_out_channel(tensor):
    t = tensor.detach().cpu().float()
    if t.ndim < 2:
        return quantize_symmetric_int8_per_tensor(t)

    reduce_dims = tuple(range(1, t.ndim))
    abs_max = t.abs().amax(dim=reduce_dims).clamp(min=1e-8)
    scale = (abs_max / 127.0).to(torch.float16)
    view_shape = [t.shape[0]] + [1] * (t.ndim - 1)
    q = torch.round(t / scale.float().view(view_shape)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def compact_float_tensor(tensor, dtype):
    t = tensor.detach().cpu()
    if t.is_floating_point():
        return t.to(dtype=dtype).contiguous()
    return t.contiguous() if not t.is_contiguous() else t.clone()


def duplicate_fuser_pair_key(key):
    if ".Fuser." not in key:
        return None
    return key.replace(".Fuser.", ".fuser.")


def is_duplicate_fuser_tensor(key, state_dict):
    pair_key = duplicate_fuser_pair_key(key)
    if pair_key is None or pair_key not in state_dict:
        return False
    value = state_dict[key]
    pair_value = state_dict[pair_key]
    return torch.is_tensor(pair_value) and torch.equal(value, pair_value)


def is_mlp_weight(key):
    return any(pattern in key for pattern in MLP_WEIGHT_PATTERNS)


def convert_checkpoint(src_path, dst_path, dtype):
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    src_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model_state = {}
    position_bias_int8 = {}
    position_bias_scale = {}
    mlp_weight_int8 = {}
    mlp_weight_scale = {}

    stats = {
        "normal_tensors": 0,
        "position_bias_quantized": 0,
        "duplicate_fuser_tensors_skipped": 0,
        "mlp_weight_quantized": 0,
        "attn_mask_skipped": 0,
        "non_tensor": 0,
    }

    for key, value in src_state.items():
        if not torch.is_tensor(value):
            model_state[key] = value
            stats["non_tensor"] += 1
            continue

        if ATTN_MASK_NAME in key:
            stats["attn_mask_skipped"] += 1
            continue

        if is_duplicate_fuser_tensor(key, src_state):
            stats["duplicate_fuser_tensors_skipped"] += 1
            continue

        if POSITION_BIAS_NAME in key:
            q, scale = quantize_symmetric_int8_per_tensor(value)
            position_bias_int8[key] = q
            position_bias_scale[key] = scale
            stats["position_bias_quantized"] += 1
            continue

        if is_mlp_weight(key):
            q, scale = quantize_symmetric_int8_per_out_channel(value)
            mlp_weight_int8[key] = q
            mlp_weight_scale[key] = scale
            stats["mlp_weight_quantized"] += 1
            continue

        model_state[key] = compact_float_tensor(value, dtype)
        stats["normal_tensors"] += 1

    out = {
        "model_state_dict": model_state,
        "position_bias_int8": position_bias_int8,
        "position_bias_scale": position_bias_scale,
        "mlp_weight_int8": mlp_weight_int8,
        "mlp_weight_scale": mlp_weight_scale,
        "light_format": {
            "version": 2,
            "normal_weight_dtype": str(dtype).replace("torch.", ""),
            "position_bias": "symmetric_int8_per_tensor_scale",
            "mlp_weight": "symmetric_int8_per_output_channel_scale",
            "mlp_bias": "kept_in_normal_weight_dtype",
            "dropped_attn_mask": "recreated by Pangu model init",
            "dropped_duplicate_fuser_tensors": "Fuser duplicates fuser when tensor values are equal",
        },
    }
    dst_dir = os.path.dirname(dst_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    torch.save(out, dst_path)
    return stats


def mb(path):
    return os.path.getsize(path) / 1024 / 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.path.join("data", "checkpoints", "model_288_latest.pth"))
    parser.add_argument(
        "--dst",
        default=os.path.join("data", "checkpoints", "model_288_light_int8_bf16_mlpint8.pth"),
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Cannot find source checkpoint: {args.src}")

    print(f"source: {args.src}")
    print(f"target: {args.dst}")
    print(f"normal tensor dtype: {args.dtype}")
    print(f"{POSITION_BIAS_NAME}: int8 + fp16 scale")
    print("MLP fc1/fc2 weight: int8 + fp16 per-output-channel scale")
    print("MLP bias and other tensors: normal tensor dtype")
    print("duplicate .Fuser. tensors: dropped when identical to .fuser.")

    stats = convert_checkpoint(args.src, args.dst, dtype)

    src_mb = mb(args.src)
    dst_mb = mb(args.dst)
    print("\nconversion stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\nsize:")
    print(f"  source: {src_mb:.2f} MB")
    print(f"  target: {dst_mb:.2f} MB")
    print(f"  compression: {src_mb / dst_mb:.2f}x")
    print(f"  saved: {(1.0 - dst_mb / src_mb) * 100:.1f}%")
    print("\nrun:")
    print(f"  LIGHT_CKPT={args.dst} python inference.py")
