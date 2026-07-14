#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

#include <limits>


namespace {

__global__ void pad_shift_window_partition_bf16_kernel(
    const uint4* input,
    uint4* output,
    int output_spatial_positions,
    int input_pressure,
    int input_latitude,
    int input_longitude,
    int padded_pressure,
    int padded_latitude,
    int padded_longitude,
    int pad_front,
    int pad_top,
    int pad_left,
    int channel_vectors,
    int shift_pl,
    int shift_lat,
    int shift_lon,
    int win_pl,
    int win_lat,
    int win_lon,
    int num_pl_windows,
    int num_lat_windows,
    int num_lon_windows
) {
    constexpr int wave_size = 64;
    constexpr int positions_per_block = 4;
    const int lane = threadIdx.x & (wave_size - 1);
    const int position_in_block = threadIdx.x / wave_size;
    const int output_spatial_index =
        blockIdx.x * positions_per_block + position_in_block;

    __shared__ int64_t input_vector_bases[positions_per_block];
    __shared__ int valid_sources[positions_per_block];

    if (lane == 0 && output_spatial_index < output_spatial_positions) {
        int remaining = output_spatial_index;
        const int local_lon = remaining % win_lon;
        remaining /= win_lon;
        const int local_lat = remaining % win_lat;
        remaining /= win_lat;
        const int local_pl = remaining % win_pl;
        remaining /= win_pl;
        const int pressure_latitude_window =
            remaining % (num_pl_windows * num_lat_windows);
        remaining /= num_pl_windows * num_lat_windows;
        const int batch_lon_window = remaining;

        const int batch_index = batch_lon_window / num_lon_windows;
        const int longitude_window = batch_lon_window % num_lon_windows;
        const int pressure_window =
            pressure_latitude_window / num_lat_windows;
        const int latitude_window =
            pressure_latitude_window % num_lat_windows;

        const int rolled_pl = pressure_window * win_pl + local_pl;
        const int rolled_lat = latitude_window * win_lat + local_lat;
        const int rolled_lon = longitude_window * win_lon + local_lon;

        int padded_source_pl = rolled_pl + shift_pl;
        int padded_source_lat = rolled_lat + shift_lat;
        int padded_source_lon = rolled_lon + shift_lon;
        if (padded_source_pl >= padded_pressure) {
            padded_source_pl -= padded_pressure;
        }
        if (padded_source_lat >= padded_latitude) {
            padded_source_lat -= padded_latitude;
        }
        if (padded_source_lon >= padded_longitude) {
            padded_source_lon -= padded_longitude;
        }

        const int input_pl = padded_source_pl - pad_front;
        const int input_lat = padded_source_lat - pad_top;
        const int input_lon = padded_source_lon - pad_left;
        const bool valid =
            input_pl >= 0 && input_pl < input_pressure
            && input_lat >= 0 && input_lat < input_latitude
            && input_lon >= 0 && input_lon < input_longitude;
        valid_sources[position_in_block] = valid ? 1 : 0;
        if (valid) {
            input_vector_bases[position_in_block] =
                (((static_cast<int64_t>(batch_index) * input_pressure + input_pl)
                  * input_latitude + input_lat) * input_longitude + input_lon)
                * channel_vectors;
        }
    }

    __syncthreads();

    if (
        output_spatial_index < output_spatial_positions
        && lane < channel_vectors
    ) {
        const int64_t output_index =
            static_cast<int64_t>(output_spatial_index) * channel_vectors + lane;
        if (valid_sources[position_in_block]) {
            output[output_index] =
                input[input_vector_bases[position_in_block] + lane];
        } else {
            output[output_index] = make_uint4(0, 0, 0, 0);
        }
    }
}

}  // namespace


torch::Tensor shift_window_partition_hip(
    torch::Tensor input,
    int64_t shift_pl,
    int64_t shift_lat,
    int64_t shift_lon,
    int64_t win_pl,
    int64_t win_lat,
    int64_t win_lon
) {
    const int64_t batch = input.size(0);
    const int64_t input_pressure = input.size(1);
    const int64_t input_latitude = input.size(2);
    const int64_t input_longitude = input.size(3);
    const int64_t channels = input.size(4);

    const int64_t padded_pressure =
        ((input_pressure + win_pl - 1) / win_pl) * win_pl;
    const int64_t padded_latitude =
        ((input_latitude + win_lat - 1) / win_lat) * win_lat;
    const int64_t padded_longitude =
        ((input_longitude + win_lon - 1) / win_lon) * win_lon;
    const int64_t pad_front = (padded_pressure - input_pressure) / 2;
    const int64_t pad_top = (padded_latitude - input_latitude) / 2;
    const int64_t pad_left = (padded_longitude - input_longitude) / 2;

    const int64_t num_pl_windows = padded_pressure / win_pl;
    const int64_t num_lat_windows = padded_latitude / win_lat;
    const int64_t num_lon_windows = padded_longitude / win_lon;

    auto output = torch::empty(
        {
            batch * num_lon_windows,
            num_pl_windows * num_lat_windows,
            win_pl,
            win_lat,
            win_lon,
            channels,
        },
        input.options()
    );

    constexpr int vector_width = 8;
    const int64_t output_spatial_positions_64 = output.numel() / channels;
    if (output_spatial_positions_64 == 0) {
        return output;
    }
    TORCH_CHECK(
        output_spatial_positions_64
            <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "spatial output is too large for the gfx936 kernel"
    );
    const int output_spatial_positions =
        static_cast<int>(output_spatial_positions_64);
    const int channel_vectors = static_cast<int>(channels / vector_width);

    constexpr int threads = 256;
    constexpr int positions_per_block = 4;
    const int blocks =
        (output_spatial_positions + positions_per_block - 1)
        / positions_per_block;
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    hipLaunchKernelGGL(
        pad_shift_window_partition_bf16_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        stream.stream(),
        reinterpret_cast<const uint4*>(input.data_ptr()),
        reinterpret_cast<uint4*>(output.data_ptr()),
        output_spatial_positions,
        static_cast<int>(input_pressure),
        static_cast<int>(input_latitude),
        static_cast<int>(input_longitude),
        static_cast<int>(padded_pressure),
        static_cast<int>(padded_latitude),
        static_cast<int>(padded_longitude),
        static_cast<int>(pad_front),
        static_cast<int>(pad_top),
        static_cast<int>(pad_left),
        channel_vectors,
        static_cast<int>(shift_pl),
        static_cast<int>(shift_lat),
        static_cast<int>(shift_lon),
        static_cast<int>(win_pl),
        static_cast<int>(win_lat),
        static_cast<int>(win_lon),
        static_cast<int>(num_pl_windows),
        static_cast<int>(num_lat_windows),
        static_cast<int>(num_lon_windows)
    );

    const hipError_t error = hipGetLastError();
    TORCH_CHECK(
        error == hipSuccess,
        "pad_shift_window_partition_bf16_kernel launch failed: ",
        hipGetErrorString(error)
    );
    return output;
}
