import types

import torch

from .runtime import get_window_extension


SUPPORTED_WINDOW_SIZE = (2, 6, 12)
SUPPORTED_CHANNELS = {192, 384}
_WINDOW_EXTENSION = get_window_extension()


def _window_reverse(windows, window_size, pressure, latitude, longitude):
    win_pl, win_lat, win_lon = window_size
    batch = int(windows.shape[0] / (longitude / win_lon))
    x = windows.view(
        batch,
        longitude // win_lon,
        pressure // win_pl,
        latitude // win_lat,
        win_pl,
        win_lat,
        win_lon,
        -1,
    )
    return (
        x.permute(0, 2, 4, 3, 5, 1, 6, 7)
        .contiguous()
        .view(batch, pressure, latitude, longitude, -1)
    )


def _crop_to_resolution(x, resolution):
    pressure, latitude, longitude = resolution
    pressure_pad = x.shape[1] - pressure
    latitude_pad = x.shape[2] - latitude
    longitude_pad = x.shape[3] - longitude
    pressure_front = pressure_pad // 2
    latitude_top = latitude_pad // 2
    longitude_left = longitude_pad // 2
    return x[
        :,
        pressure_front : pressure_front + pressure,
        latitude_top : latitude_top + latitude,
        longitude_left : longitude_left + longitude,
        :,
    ]


def _fused_window_forward(self, x):
    pressure, height, width = self.input_resolution
    batch, _, channels = x.shape

    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or tuple(self.window_size) != SUPPORTED_WINDOW_SIZE
        or channels not in SUPPORTED_CHANNELS
    ):
        return self._dcu_original_forward(x)

    shortcut = x
    x = self.norm1(x)
    x = x.view(batch, pressure, height, width, channels)
    if not x.is_contiguous():
        return self._dcu_original_forward(shortcut)

    shift_pl, shift_height, shift_width = (
        tuple(self.shift_size) if self.use_roll else (0, 0, 0)
    )
    win_pl, win_height, win_width = self.window_size
    x_windows = _WINDOW_EXTENSION.shift_window_partition(
        x,
        shift_pl,
        shift_height,
        shift_width,
        win_pl,
        win_height,
        win_width,
    )
    padded_pressure = ((pressure + win_pl - 1) // win_pl) * win_pl
    padded_height = ((height + win_height - 1) // win_height) * win_height
    padded_width = ((width + win_width - 1) // win_width) * win_width

    x_windows = x_windows.view(
        x_windows.shape[0],
        x_windows.shape[1],
        win_pl * win_height * win_width,
        channels,
    )
    attn_windows = self.attn(x_windows, mask=self.attn_mask)
    attn_windows = attn_windows.view(
        attn_windows.shape[0],
        attn_windows.shape[1],
        win_pl,
        win_height,
        win_width,
        channels,
    )

    shifted_x = _window_reverse(
        attn_windows,
        self.window_size,
        padded_pressure,
        padded_height,
        padded_width,
    )
    if self.use_roll:
        x = torch.roll(
            shifted_x,
            shifts=(shift_pl, shift_height, shift_width),
            dims=(1, 2, 3),
        )
    else:
        x = shifted_x

    x = _crop_to_resolution(x, self.input_resolution)
    x = x.reshape(batch, pressure * height * width, channels)
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def enable_fused_window_partition(model):
    """Patch compatible EarthTransformer3DBlock instances in-place."""
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "EarthTransformer3DBlock":
            continue
        required = (
            "input_resolution",
            "window_size",
            "shift_size",
            "use_roll",
            "norm1",
            "norm2",
            "attn",
            "mlp",
            "drop_path",
        )
        if not all(hasattr(module, name) for name in required):
            continue
        if hasattr(module, "_dcu_original_forward"):
            continue
        module._dcu_original_forward = module.forward
        module.forward = types.MethodType(_fused_window_forward, module)
        patched += 1
    return patched
