#include <torch/extension.h>


torch::Tensor shift_window_partition_hip(
    torch::Tensor input,
    int64_t shift_pl,
    int64_t shift_lat,
    int64_t shift_lon,
    int64_t win_pl,
    int64_t win_lat,
    int64_t win_lon
);


torch::Tensor shift_window_partition(
    torch::Tensor input,
    int64_t shift_pl,
    int64_t shift_lat,
    int64_t shift_lon,
    int64_t win_pl,
    int64_t win_lat,
    int64_t win_lon
) {
    TORCH_CHECK(input.is_cuda(), "input must be on a CUDA/HIP/DCU device");
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16,
        "input must have torch.bfloat16 dtype"
    );
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 5, "input must have shape [B, Pl, Lat, Lon, C]");
    TORCH_CHECK(win_pl > 0 && win_lat > 0 && win_lon > 0, "window sizes must be positive");
    TORCH_CHECK(
        input.size(4) % 8 == 0,
        "channels must be divisible by 8 for vectorized BF16 copies"
    );
    return shift_window_partition_hip(
        input,
        shift_pl,
        shift_lat,
        shift_lon,
        win_pl,
        win_lat,
        win_lon
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "shift_window_partition",
        &shift_window_partition,
        "Fused symmetric pad, negative roll, and 3D window partition"
    );
}
