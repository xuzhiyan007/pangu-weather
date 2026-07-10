import argparse
import csv
import json
import os
import random

import h5py
import numpy as np
import yaml


TARGET_VARS = [
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
]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(data_dir):
    nested = os.path.join(data_dir, "data")
    return nested if os.path.isdir(nested) else data_dir


def find_metadata(data_dir):
    candidates = [
        os.path.join(data_dir, "metadata.json"),
        os.path.join(os.path.dirname(data_dir), "metadata.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"metadata.json not found near {data_dir}")


def get_indices(all_names, names):
    missing = [name for name in names if name not in all_names]
    if missing:
        raise KeyError(f"Missing variables in metadata.json: {missing}")
    return [all_names.index(name) for name in names]


def list_consecutive_pairs(data_root, years, max_pairs, seed):
    pairs = []
    for year in years:
        year_dir = os.path.join(data_root, str(year))
        files = sorted(
            os.path.join(year_dir, name)
            for name in os.listdir(year_dir)
            if name.endswith(".h5")
        )
        pairs.extend(zip(files[:-1], files[1:]))

    random.seed(seed)
    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    return sorted(pairs)


def read_selected_fields(path, indices, spatial_stride):
    indices = np.asarray(indices, dtype=np.int64)

    # h5py fancy indexing requires increasing-order indices.
    order = np.argsort(indices)
    sorted_indices = indices[order]

    with h5py.File(path, "r") as f:
        data_sorted = f["fields"][sorted_indices.tolist(), ::spatial_stride, ::spatial_stride]

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    data = data_sorted[inverse_order]

    return data.reshape(len(indices), -1).astype(np.float64, copy=False)


def write_matrix_csv(path, row_names, col_names, matrix):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_variable", *col_names])
        for name, row in zip(row_names, matrix):
            writer.writerow([name, *[f"{v:.6f}" for v in row]])


def write_topk(path, input_names, target_names, corr, topk):
    mean_abs = np.mean(np.abs(corr), axis=1)
    overall = np.argsort(-mean_abs)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Overall input-variable ranking by mean abs correlation to target15\n")
        for rank, idx in enumerate(overall, 1):
            f.write(f"{rank:02d}. {input_names[idx]:35s} mean_abs_corr={mean_abs[idx]:.6f}\n")

        f.write("\nPer-target top variables\n")
        for j, target in enumerate(target_names):
            f.write(f"\n[target] {target}\n")
            order = np.argsort(-np.abs(corr[:, j]))
            for rank, idx in enumerate(order[:topk], 1):
                f.write(
                    f"{rank:02d}. {input_names[idx]:35s} "
                    f"corr={corr[idx, j]: .6f} abs={abs(corr[idx, j]):.6f}\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument("--spatial-stride", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out-dir", default="result/correlation_target15")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = cfg["datapipe"]["dataset"]

    data_dir = dataset["data_dir"]
    data_root = resolve_data_root(data_dir)
    train_years = dataset["train_ratio"]
    input_names = dataset["channels"]

    metadata_path = find_metadata(data_dir)
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_names = json.load(f)["variables"]

    input_indices = get_indices(all_names, input_names)
    target_indices = get_indices(all_names, TARGET_VARS)

    os.makedirs(args.out_dir, exist_ok=True)
    indices_path = os.path.join(args.out_dir, "selected_indices.json")
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata_path": metadata_path,
                "data_root": data_root,
                "train_years": train_years,
                "input_variables": [
                    {"name": name, "index": idx}
                    for name, idx in zip(input_names, input_indices)
                ],
                "target_variables": [
                    {"name": name, "index": idx}
                    for name, idx in zip(TARGET_VARS, target_indices)
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"metadata: {metadata_path}")
    print(f"data_root: {data_root}")
    print(f"input indices: {input_indices}")
    print(f"target indices: {target_indices}")
    print(f"saved indices: {indices_path}")

    pairs = list_consecutive_pairs(data_root, train_years, args.max_pairs, args.seed)
    print(f"selected file pairs: {len(pairs)}")

    sx = np.zeros(len(input_indices), dtype=np.float64)
    sy = np.zeros(len(target_indices), dtype=np.float64)
    sx2 = np.zeros(len(input_indices), dtype=np.float64)
    sy2 = np.zeros(len(target_indices), dtype=np.float64)
    sxy = np.zeros((len(input_indices), len(target_indices)), dtype=np.float64)
    n = 0

    for k, (input_path, target_path) in enumerate(pairs, 1):
        x = read_selected_fields(input_path, input_indices, args.spatial_stride)
        y = read_selected_fields(target_path, target_indices, args.spatial_stride)

        sx += x.sum(axis=1)
        sy += y.sum(axis=1)
        sx2 += (x * x).sum(axis=1)
        sy2 += (y * y).sum(axis=1)
        sxy += x @ y.T
        n += x.shape[1]

        if k % 10 == 0 or k == len(pairs):
            print(f"processed {k}/{len(pairs)} pairs, points={n}")

    cov = sxy - np.outer(sx, sy) / n
    vx = sx2 - sx * sx / n
    vy = sy2 - sy * sy / n
    corr = cov / (np.sqrt(np.outer(vx, vy)) + 1e-12)

    matrix_path = os.path.join(args.out_dir, "input69_to_target15_pearson.csv")
    topk_path = os.path.join(args.out_dir, "topk_related_inputs.txt")

    write_matrix_csv(matrix_path, input_names, TARGET_VARS, corr)
    write_topk(topk_path, input_names, TARGET_VARS, corr, args.topk)

    print(f"saved: {matrix_path}")
    print(f"saved: {topk_path}")


if __name__ == "__main__":
    main()