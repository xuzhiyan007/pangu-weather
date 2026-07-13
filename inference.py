"""Standalone inference for an INT8 Pangu model."""

import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
from pangu_patchsize import PanguPatchSize as Pangu


POSITION_BIAS_NAME = "earth_position_bias_table"
DEFAULT_MODEL_DEPTH = (2, 6, 6, 2)
DEFAULT_PATCH_SIZE = (2, 10, 10)
DEFAULT_CHECKPOINT = os.path.join(
    "data", "checkpoints", "model_2_10_10_light_int8_bf16.pth"
)


def get_stats(data_dir, channels):
    with open(os.path.join(data_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    all_variables = metadata["variables"]
    channel_indices = [all_variables.index(variable) for variable in channels]
    stats_dir = os.path.join(data_dir, "stats")
    means = np.load(os.path.join(stats_dir, "global_means.npy"))
    stds = np.load(os.path.join(stats_dir, "global_stds.npy"))
    return means[:, channel_indices, :, :], stds[:, channel_indices, :, :]


def expand_scale(scale, quantized, group_size=0):
    scale = scale.float()
    if group_size > 0:
        if quantized.ndim != 2 or scale.ndim != 2:
            raise ValueError(
                "Grouped INT8 requires 2D weight and scale tensors, got "
                f"{tuple(quantized.shape)} and {tuple(scale.shape)}"
            )
        return scale.repeat_interleave(group_size, dim=1)[:, : quantized.shape[1]]
    if scale.ndim == 1 and quantized.ndim >= 2:
        if scale.numel() != quantized.shape[0]:
            raise ValueError(
                f"INT8 scale length {scale.numel()} does not match "
                f"output dimension {quantized.shape[0]}"
            )
        return scale.view([quantized.shape[0]] + [1] * (quantized.ndim - 1))
    return scale


def restore_int8_group(
    state_dict,
    checkpoint,
    weight_field,
    scale_field,
    infer_dtype,
    group_size=0,
):
    quantized_weights = checkpoint.get(weight_field, {})
    scales = checkpoint.get(scale_field, {})
    for key, quantized in quantized_weights.items():
        if key not in scales:
            raise KeyError(f"Missing INT8 scale for {key}")
        scale = expand_scale(scales[key], quantized, group_size=group_size)
        state_dict[key] = (quantized.float() * scale).to(infer_dtype)


def restore_fuser_aliases(state_dict):
    additions = {}
    for key, value in list(state_dict.items()):
        if ".fuser." in key:
            pair_key = key.replace(".fuser.", ".Fuser.")
        elif ".Fuser." in key:
            pair_key = key.replace(".Fuser.", ".fuser.")
        else:
            continue
        if pair_key not in state_dict:
            additions[pair_key] = value.clone()
    state_dict.update(additions)


def unpack_checkpoint(checkpoint, infer_dtype):
    light_format = checkpoint.get("light_format", {})
    checkpoint_depth = tuple(
        int(value) for value in light_format.get("model_depth", DEFAULT_MODEL_DEPTH)
    )
    checkpoint_patch_size = tuple(
        int(value)
        for value in light_format.get("model_patch_size", DEFAULT_PATCH_SIZE)
    )
    if light_format.get("int4_scope") not in (None, "none"):
        raise ValueError("inference.py does not accept INT4 checkpoints")

    state_dict = dict(checkpoint.get("model_state_dict", checkpoint))
    group_size = int(light_format.get("mlp_group_size", 0))
    restore_int8_group(
        state_dict,
        checkpoint,
        "position_bias_int8",
        "position_bias_scale",
        infer_dtype,
    )
    restore_int8_group(
        state_dict,
        checkpoint,
        "mlp_weight_int8",
        "mlp_weight_scale",
        infer_dtype,
        group_size=group_size,
    )

    unresolved = [
        key
        for key, value in state_dict.items()
        if torch.is_tensor(value) and value.dtype in (torch.int8, torch.uint8)
    ]
    if unresolved:
        raise ValueError(
            "Quantized tensors have no supported scale: " + ", ".join(unresolved[:10])
        )
    restore_fuser_aliases(state_dict)
    return state_dict, light_format, checkpoint_depth, checkpoint_patch_size


def load_static_surface_mask(cfg_data, device, dtype):
    static_dir = cfg_data.dataset.static_dir
    land_mask = torch.from_numpy(
        np.load(os.path.join(static_dir, "land_mask.npy")).astype(np.float32)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(static_dir, "soil_type.npy")).astype(np.float32)
    )
    topography = torch.from_numpy(
        np.load(os.path.join(static_dir, "topography.npy")).astype(np.float32)
    )
    topography = (
        topography - topography.mean()
    ) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0)
    return surface_mask.unsqueeze(0).to(device=device, dtype=dtype)


def main():
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_path = os.path.join(current_path, "conf", "config.yaml")
    cfg = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")

    checkpoint_path = os.environ.get("LIGHT_CKPT", DEFAULT_CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Cannot find light checkpoint: {checkpoint_path}. "
            "Run python convert.py first."
        )

    device = torch.device("cuda:0")
    infer_dtype = torch.bfloat16
    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()
    surface_mask = load_static_surface_mask(cfg_data, device, infer_dtype)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    (
        model_state,
        light_format,
        checkpoint_depth,
        checkpoint_patch_size,
    ) = unpack_checkpoint(checkpoint, infer_dtype)
    config_depth = tuple(int(value) for value in cfg.depth)
    config_patch_size = tuple(int(value) for value in cfg.patch_size)
    if config_depth != checkpoint_depth:
        raise ValueError(
            f"conf/config.yaml model.depth is {config_depth}, but checkpoint "
            f"depth is {checkpoint_depth}"
        )
    if config_patch_size != checkpoint_patch_size:
        raise ValueError(
            f"conf/config.yaml model.patch_size is {config_patch_size}, but "
            f"checkpoint patch size is {checkpoint_patch_size}"
        )
    model = Pangu(
        img_size=cfg_data.dataset.img_size,
        patch_size=checkpoint_patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        depth=checkpoint_depth,
    ).to(device=device, dtype=infer_dtype)
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    bad_missing = [key for key in missing_keys if "attn_mask" not in key]
    if bad_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the configured model. "
            f"missing={bad_missing[:20]}, unexpected={unexpected_keys[:20]}"
        )
    model.eval()

    output_dir = os.path.join("result", "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}")
    print(f"model depth: {list(checkpoint_depth)}")
    print(f"patch size: {list(checkpoint_patch_size)}")
    print(f"infer dtype: {infer_dtype}")
    print(f"int8 storage scope: {light_format.get('int8_scope', 'unknown')}")
    if missing_keys:
        print(f"recreated {len(missing_keys)} attn_mask buffers from model init")
    print("samples will be generated to './result/output/'")

    time_list = []
    generated_count = 0
    first = True
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]
            filename = data[4][-1][0]
            if first:
                first = False
                print(f"  invar shape: {list(invar.shape)} -> [batch, channels, H, W]")
                print(f"  the first filename: {filename}")

            invar_surface = invar[:, :4].to(device=device, dtype=infer_dtype)
            invar_upper_air = invar[:, 4:].to(device=device, dtype=infer_dtype)
            batch_mask = surface_mask.expand(invar_surface.shape[0], -1, -1, -1)
            model_input = torch.concat(
                [invar_surface, batch_mask, invar_upper_air], dim=1
            )

            # AI4S timing boundary: keep the model call inside this block only.
            start_time = time.perf_counter()
            out_surface, out_upper_air = model(model_input)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_list.append(end_time - start_time)

            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
            # Keep the same order as inference.py: transfer BF16 outputs to
            # CPU first, then cast to FP32 on CPU. Calling .float().cpu()
            # would create large transient FP32 tensors on the GPU.
            out_surface_cpu = out_surface.cpu().float()
            out_upper_air_cpu = out_upper_air.cpu().float()

            del (
                invar_surface,
                invar_upper_air,
                batch_mask,
                model_input,
                out_surface,
                out_upper_air,
            )
            torch.cuda.empty_cache()

            prediction = torch.concat(
                [out_surface_cpu, out_upper_air_cpu], dim=1
            ).numpy()
            prediction = prediction * stds + means
            np.save(os.path.join(output_dir, f"{filename}.npy"), prediction)
            generated_count += 1

            del out_surface_cpu, out_upper_air_cpu, prediction

    if generated_count != len(test_dataloader):
        raise RuntimeError(
            f"Generated {generated_count} files, expected {len(test_dataloader)}"
        )
    with open(os.path.join("result", "time_record.json"), "w", encoding="utf-8") as f:
        json.dump(time_list, f, ensure_ascii=False, indent=4)
    print(f"generated {generated_count} prediction file(s)")


if __name__ == "__main__":
    main()
