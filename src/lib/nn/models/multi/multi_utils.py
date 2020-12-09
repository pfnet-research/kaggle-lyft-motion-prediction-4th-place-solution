from typing import Dict, Tuple


def calc_in_out_channels(cfg: Dict, num_modes: int = 3) -> Tuple[int, int, int, int]:
    # DEPRECATED. in_channels calculation only work for "py_semantic" rasterizer...
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    in_channels = 3 + num_history_channels
    future_len = cfg["model_params"]["future_num_frames"]
    num_targets = 2 * future_len
    num_preds = num_targets * num_modes
    num_modes = num_modes
    out_dim = num_preds + num_modes
    return in_channels, out_dim, num_preds, future_len


def calc_out_channels(cfg: Dict, num_modes: int = 3) -> Tuple[int, int, int]:
    future_len = cfg["model_params"]["future_num_frames"]
    num_targets = 2 * future_len
    num_preds = num_targets * num_modes
    num_modes = num_modes
    out_dim = num_preds + num_modes
    return out_dim, num_preds, future_len
