import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from onescience.datapipes.climate import ERA5Datapipe
from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams


POSITION_BIAS_NAME = "earth_position_bias_table"


def get_stats(data_dir, channels):
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    all_variables = metadata["variables"]
    channel_indices = [all_variables.index(v) for v in channels]
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    return mu[:, channel_indices, :, :], std[:, channel_indices, :, :]


def find_checkpoint(checkpoint_dir):
    candidates = [
        os.environ.get("LIGHT_CKPT"),
        os.path.join(checkpoint_dir, "model_light_int8_bf16_mlpint8.pth"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def dequantize_tensor(q, scale, zero_point=None, dtype=torch.bfloat16):
    x = q.float()
    if zero_point is not None:
        x = x - zero_point.float()
    if torch.is_tensor(scale) and scale.ndim == 1 and q.ndim >= 2 and scale.numel() == q.shape[0]:
        scale = scale.float().view([q.shape[0]] + [1] * (q.ndim - 1))
    else:
        scale = scale.float()
    x = x * scale
    return x.to(dtype)


def restore_fuser_aliases(state_dict):
    additions = {}
    for key, value in list(state_dict.items()):
        if POSITION_BIAS_NAME not in key:
            continue
        if ".fuser." in key:
            pair_key = key.replace(".fuser.", ".Fuser.")
        elif ".Fuser." in key:
            pair_key = key.replace(".Fuser.", ".fuser.")
        else:
            continue
        if pair_key not in state_dict:
            additions[pair_key] = value.clone()
    state_dict.update(additions)


def restore_quantized_group(state_dict, ckpt, value_name, scale_name, zero_point_name, infer_dtype):
    quantized = ckpt.get(value_name, {})
    scales = ckpt.get(scale_name, {})
    zero_points = ckpt.get(zero_point_name, {})
    for key, q in quantized.items():
        if key not in scales:
            raise KeyError(f"Missing int8 scale for {key}")
        state_dict[key] = dequantize_tensor(
            q,
            scales[key],
            zero_points.get(key),
            dtype=infer_dtype,
        )


def unpack_light_checkpoint(ckpt, infer_dtype=torch.bfloat16):
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = dict(state_dict)

    restore_quantized_group(
        state_dict,
        ckpt,
        "position_bias_int8",
        "position_bias_scale",
        "position_bias_zero_point",
        infer_dtype,
    )
    restore_quantized_group(
        state_dict,
        ckpt,
        "mlp_weight_int8",
        "mlp_weight_scale",
        "mlp_weight_zero_point",
        infer_dtype,
    )

    scale_suffixes = (".scale", "_scale")
    flat_scales = {
        key: value
        for key, value in list(state_dict.items())
        if POSITION_BIAS_NAME in key and key.endswith(scale_suffixes)
    }
    for scale_key, scale in flat_scales.items():
        for suffix in scale_suffixes:
            if scale_key.endswith(suffix):
                bias_key = scale_key[: -len(suffix)]
                break
        q = state_dict.get(bias_key)
        if q is not None and getattr(q, "dtype", None) == torch.int8:
            state_dict[bias_key] = dequantize_tensor(q, scale, dtype=infer_dtype)
        state_dict.pop(scale_key, None)

    for key, value in list(state_dict.items()):
        if getattr(value, "dtype", None) == torch.int8:
            raise ValueError(f"{key} is int8 but no scale was found for dequantization")

    restore_fuser_aliases(state_dict)
    return state_dict


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    device = torch.device("cuda:0")
    infer_dtype = torch.bfloat16

    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False)
    test_dataloader = datapipe.test_dataloader()

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(device=device, dtype=infer_dtype)
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)

    ckpt_path = find_checkpoint(cfg.checkpoint_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = unpack_light_checkpoint(ckpt, infer_dtype=infer_dtype)

    model = Pangu(
        img_size=cfg_data.dataset.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
    ).to(device=device, dtype=infer_dtype)
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    bad_missing = [key for key in missing_keys if "attn_mask" not in key]
    if bad_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint did not match model. "
            f"bad_missing={bad_missing[:20]}, unexpected={unexpected_keys[:20]}"
        )
    if missing_keys:
        print(f"recreated {len(missing_keys)} attn_mask buffers from model init")
    model.eval()

    os.makedirs("result/output/", exist_ok=True)
    print(f"checkpoint: {ckpt_path}")
    print(f"infer dtype: {infer_dtype}")
    print("extra int8 storage: earth_position_bias_table + MLP fc1/fc2 weight")
    print("samples will be generated to './result/output/'")

    time_list = []
    first = True
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]
            outvar = data[1]
            filename = data[4][-1][0]
            if first:
                first = False
                print(f"  invar  shape: {list(invar.shape)} -> [batch, channels, H, W]")
                print(f"  outvar shape: {list(outvar.shape)} -> [batch, channels, H, W]")
                print(f"  the first filename: {filename}")

            invar_surface = invar[:, :4, :, :].to(device=device, dtype=infer_dtype)
            invar_upper_air = invar[:, 4:, :, :].to(device=device, dtype=infer_dtype)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

            #----------------------AI4S(time measurement, do not move)---------------------------
            start_time = time.perf_counter()
            out_surface, out_upper_air = model(invar)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_list.append(end_time - start_time)
            #------------------------------------------------------------------------------------

            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
            out_surface_cpu = out_surface.cpu().float()
            out_upper_air_cpu = out_upper_air.cpu().float()

            del out_surface, out_upper_air, invar, invar_surface, invar_upper_air
            torch.cuda.empty_cache()

            pred_var = torch.concat([out_surface_cpu, out_upper_air_cpu], dim=1).numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)

            del out_surface_cpu, out_upper_air_cpu, pred_var

        with open("result/time_record.json", "w", encoding="utf-8") as f:
            json.dump(time_list, f, ensure_ascii=False, indent=4)
