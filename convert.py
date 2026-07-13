"""Convert the [2, 6, 4, 2] pruned checkpoint with convert1's INT8 scheme."""

import argparse
import os

import torch


POSITION_BIAS_NAME = "earth_position_bias_table"
ATTN_MASK_NAME = "attn_mask"


def mb(path):
    return os.path.getsize(path) / 1024 / 1024


def quantize_int8_per_tensor(tensor):
    value = tensor.detach().cpu().float()
    scale = (value.abs().max().clamp(min=1e-8) / 127.0).to(torch.float16)
    quantized = torch.round(value / scale.float()).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale


def quantize_int8_per_output(tensor):
    value = tensor.detach().cpu().float()
    reduce_dims = tuple(range(1, value.ndim))
    scale = (
        value.abs().amax(dim=reduce_dims).clamp(min=1e-8) / 127.0
    ).to(torch.float16)
    scale_view = scale.float().view([value.shape[0]] + [1] * (value.ndim - 1))
    quantized = torch.round(value / scale_view).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


def quantize_int8_per_group(tensor, group_size):
    value = tensor.detach().cpu().float()
    if value.ndim != 2 or group_size <= 0:
        return quantize_int8_per_output(value)
    out_features, in_features = value.shape
    group_count = (in_features + group_size - 1) // group_size
    padded_features = group_count * group_size
    if padded_features != in_features:
        value = torch.cat(
            [
                value,
                torch.zeros(
                    out_features,
                    padded_features - in_features,
                    dtype=value.dtype,
                ),
            ],
            dim=1,
        )
    grouped = value.reshape(out_features, group_count, group_size)
    scale = (
        grouped.abs().amax(dim=2).clamp(min=1e-8) / 127.0
    ).to(torch.float16)
    quantized = torch.round(grouped / scale.float().unsqueeze(-1))
    quantized = quantized.clamp(-127, 127).to(torch.int8)
    quantized = quantized.reshape(out_features, padded_features)[:, :in_features]
    return quantized.contiguous(), scale.contiguous()


def quantization_error_sums(tensor, quantized, scale, group_size=0):
    original = tensor.detach().cpu().float()
    if group_size > 0:
        expanded_scale = scale.float().repeat_interleave(group_size, dim=1)
        restored = quantized.float() * expanded_scale[:, :quantized.shape[1]]
    elif scale.ndim == 1 and quantized.ndim >= 2:
        scale_view = scale.float().view(
            [quantized.shape[0]] + [1] * (quantized.ndim - 1)
        )
        restored = quantized.float() * scale_view
    else:
        restored = quantized.float() * scale.float()
    return (
        (original - restored).square().sum().item(),
        original.square().sum().item(),
    )


def compact_tensor(tensor, dtype):
    value = tensor.detach().cpu()
    if value.is_floating_point():
        return value.to(dtype=dtype).contiguous()
    return value.contiguous() if not value.is_contiguous() else value.clone()


def is_duplicate_fuser_tensor(key, state_dict):
    if ".Fuser." not in key:
        return False
    pair_key = key.replace(".Fuser.", ".fuser.")
    if pair_key not in state_dict:
        return False
    pair_value = state_dict[pair_key]
    return torch.is_tensor(pair_value) and torch.equal(state_dict[key], pair_value)


def convert_checkpoint(
    src_path,
    dst_path,
    dtype,
    position_bias_dtype="int8",
    linear_group_size=0,
):
    checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint)

    model_state = {}
    position_bias_int8 = {}
    position_bias_scale = {}
    matrix_int8 = {}
    matrix_scale = {}
    position_error_sq = 0.0
    position_signal_sq = 0.0
    matrix_error_sq = 0.0
    matrix_signal_sq = 0.0
    stats = {
        "normal_tensors": 0,
        "position_bias_quantized": 0,
        "linear_weight_quantized": 0,
        "duplicate_fuser_tensors_skipped": 0,
        "attn_mask_skipped": 0,
        "non_tensor": 0,
    }

    for key, value in source_state.items():
        if not torch.is_tensor(value):
            model_state[key] = value
            stats["non_tensor"] += 1
            continue
        if ATTN_MASK_NAME in key:
            stats["attn_mask_skipped"] += 1
            continue
        if is_duplicate_fuser_tensor(key, source_state):
            stats["duplicate_fuser_tensors_skipped"] += 1
            continue
        if POSITION_BIAS_NAME in key and position_bias_dtype == "int8":
            quantized, scale = quantize_int8_per_tensor(value)
            position_bias_int8[key] = quantized
            position_bias_scale[key] = scale
            error_sq, signal_sq = quantization_error_sums(value, quantized, scale)
            position_error_sq += error_sq
            position_signal_sq += signal_sq
            stats["position_bias_quantized"] += 1
            continue
        if key.endswith(".weight") and value.ndim == 2:
            quantized, scale = quantize_int8_per_group(value, linear_group_size)
            matrix_int8[key] = quantized
            matrix_scale[key] = scale
            error_sq, signal_sq = quantization_error_sums(
                value, quantized, scale, linear_group_size
            )
            matrix_error_sq += error_sq
            matrix_signal_sq += signal_sq
            stats["linear_weight_quantized"] += 1
            continue
        model_state[key] = compact_tensor(value, dtype)
        stats["normal_tensors"] += 1

    stats["position_bias_relative_rmse"] = (
        (position_error_sq / position_signal_sq) ** 0.5
        if position_signal_sq
        else 0.0
    )
    stats["linear_weight_relative_rmse"] = (
        (matrix_error_sq / matrix_signal_sq) ** 0.5
        if matrix_signal_sq
        else 0.0
    )
    output = {
        "model_state_dict": model_state,
        "position_bias_int8": position_bias_int8,
        "position_bias_scale": position_bias_scale,
        "mlp_weight_int8": matrix_int8,
        "mlp_weight_scale": matrix_scale,
        "light_format": {
            "version": 3,
            "normal_weight_dtype": str(dtype).replace("torch.", ""),
            "position_bias": (
                "symmetric_int8_per_tensor_scale"
                if position_bias_dtype == "int8"
                else str(dtype).replace("torch.", "")
            ),
            "matrix_weight": (
                f"symmetric_int8_per_{linear_group_size}_input_group_scale"
                if linear_group_size > 0
                else "symmetric_int8_per_output_channel_scale"
            ),
            "mlp_group_size": linear_group_size,
            "int8_scope": "all_2d_weights",
            "bias_norm_and_conv": "kept_in_normal_weight_dtype",
            "dropped_attn_mask": "recreated by Pangu model init",
            "dropped_duplicate_fuser_tensors": (
                "Fuser duplicates fuser when tensor values are equal"
            ),
            "model_depth": [2, 6, 4, 2],
        },
    }
    target_dir = os.path.dirname(dst_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    temporary_path = dst_path + ".tmp"
    torch.save(output, temporary_path)
    os.replace(temporary_path, dst_path)
    return stats


DEFAULT_SOURCE = os.path.join(
    "data", "checkpoints", "model_299_299_latest.pth"
)
DEFAULT_TARGET = os.path.join(
    "data", "checkpoints", "model_299_299_light_int8_bf16.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply convert1's all-2D-weight INT8 storage format to the "
            "depth=[2,6,4,2] pruned model."
        )
    )
    parser.add_argument("--src", default=DEFAULT_SOURCE)
    parser.add_argument("--dst", default=DEFAULT_TARGET)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--position-bias-dtype",
        choices=["int8", "normal"],
        default="int8",
    )
    parser.add_argument(
        "--linear-group-size",
        type=int,
        default=0,
        help="0 uses one INT8 scale per output row, matching convert1's default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.linear_group_size < 0:
        raise ValueError("--linear-group-size must be 0 or a positive integer")
    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Cannot find source checkpoint: {args.src}")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"source pruned checkpoint: {args.src}")
    print(f"target light checkpoint: {args.dst}")
    print("expected model depth: [2, 6, 4, 2]")
    print(f"normal tensor dtype: {args.dtype}")
    print(f"{POSITION_BIAS_NAME}: {args.position_bias_dtype}")
    if args.linear_group_size > 0:
        print(
            "all 2D weights: int8 + fp16 scale per "
            f"{args.linear_group_size} inputs"
        )
    else:
        print("all 2D weights: int8 + fp16 per-output-channel scale")
    print("bias, normalization, and convolution tensors: normal tensor dtype")

    stats = convert_checkpoint(
        args.src,
        args.dst,
        dtype,
        position_bias_dtype=args.position_bias_dtype,
        linear_group_size=args.linear_group_size,
    )

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
    print(f"  LIGHT_CKPT={args.dst} python inference1.py")


if __name__ == "__main__":
    main()
