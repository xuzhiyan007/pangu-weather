import hashlib
import os
from pathlib import Path
import tempfile
import time

import torch


_EXTENSION = None


def _source_digest(paths):
    digest = hashlib.sha256()
    digest.update(torch.__version__.encode("utf-8"))
    digest.update(str(torch.version.hip).encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def get_window_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if torch.version.hip is None:
        raise RuntimeError("The online DCU extension requires a HIP PyTorch build")

    os.environ.setdefault("ROCM_HOME", "/opt/dtk")
    os.environ.setdefault("HIP_HOME", "/opt/dtk")
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx936")
    os.environ.setdefault("MAX_JOBS", "4")

    # Import only after the DTK environment variables have been established.
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent
    sources = [
        source_dir / "window_binding.cpp",
        source_dir / "window_partition_kernel.cu",
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Missing online extension source: {source}")

    digest = _source_digest(sources)
    extension_name = f"pangu_dcu_window_{digest}"
    build_root = Path(
        os.environ.get(
            "PANGU_DCU_BUILD_ROOT",
            os.path.join(tempfile.gettempdir(), "pangu_dcu_extensions"),
        )
    )
    build_directory = build_root / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)

    verbose = os.environ.get("DCU_OP_BUILD_VERBOSE", "0") != "0"
    start_time = time.perf_counter()
    _EXTENSION = load(
        name=extension_name,
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start_time
    print(
        f"online DCU window extension ready: {extension_name}, "
        f"build/load time {elapsed:.2f}s"
    )
    return _EXTENSION
