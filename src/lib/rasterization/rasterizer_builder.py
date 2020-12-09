from copy import deepcopy

from l5kit.data import DataManager
from l5kit.rasterization import Rasterizer, build_rasterizer

from lib.rasterization.agent_type_box_rasterizer import AgentTypeBoxRasterizer
from lib.rasterization.augmented_box_rasterizer import AugmentedBoxRasterizer
from lib.rasterization.channel_semantic_rasterizer import ChannelSemanticRasterizer
from lib.rasterization.channel_semantic_tl_rasterizer import ChannelSemanticTLRasterizer
from lib.rasterization.combined_rasterizer import CombinedRasterizer
from lib.rasterization.tl_semantic_rasterizer import TLSemanticRasterizer
from lib.rasterization.tuned_semantic_rasterizer import TunedSemanticRasterizer
from lib.rasterization.velocity_rasterizer import VelocityBoxRasterizer
from lib.rasterization.tuned_box_rasterizer import TunedBoxRasterizer


def build_one_rasterizer(map_type: str, cfg: dict, data_manager: DataManager, eval: bool = False) -> Rasterizer:
    cfg_ = deepcopy(cfg)
    cfg_["raster_params"]["map_type"] = map_type
    if map_type == "channel_semantic":
        return ChannelSemanticRasterizer.from_cfg(cfg_, data_manager)
    elif map_type == "channel_semantic_tl":
        return ChannelSemanticTLRasterizer.from_cfg(cfg_, data_manager)
    elif map_type == "velocity_box":
        return VelocityBoxRasterizer.from_cfg(cfg_, data_manager)
    elif map_type == "agent_type_box":
        return AgentTypeBoxRasterizer.from_cfg(cfg_, data_manager)
    elif map_type == "augmented_box":
        return AugmentedBoxRasterizer.from_cfg(cfg_, data_manager, eval=eval)
    elif map_type == "tl_semantic":
        return TLSemanticRasterizer.from_cfg(cfg_, data_manager)
    elif map_type == "tuned_box":
        return TunedBoxRasterizer.from_cfg(cfg_, data_manager)
    elif map_type == "tuned_semantic":
        return TunedSemanticRasterizer.from_cfg(cfg_, data_manager)
    else:
        # Use Original l5kit rasterizer
        rasterizer = build_rasterizer(cfg_, data_manager)
        if map_type == "py_satellite":
            rasterizer.raster_channels = (rasterizer.history_num_frames + 1) * 2 + 3
        elif map_type == "satellite_debug":
            rasterizer.raster_channels = 3
        elif map_type == "py_semantic":
            rasterizer.raster_channels = (rasterizer.history_num_frames + 1) * 2 + 3
        elif map_type == "semantic_debug":
            rasterizer.raster_channels = 3
        elif map_type == "box_debug":
            rasterizer.raster_channels = (rasterizer.history_num_frames + 1) * 2
        elif map_type == "stub_debug":
            history_num_frames = cfg_["model_params"]["history_num_frames"]
            rasterizer.raster_channels = history_num_frames * 2
        return rasterizer


def build_custom_rasterizer(cfg: dict, data_manager: DataManager, eval: bool = False) -> Rasterizer:
    raster_cfg = cfg["raster_params"]
    map_type = raster_cfg["map_type"]

    map_type_list = map_type.split("+")
    rasterizer_list = [build_one_rasterizer(map_type, cfg, data_manager, eval=eval)
                       for map_type in map_type_list]
    if len(rasterizer_list) == 1:
        # Only 1 rasterizer used.
        rasterizer = rasterizer_list[0]
    else:
        # If more than 2, use combined rasterizer.
        rasterizer = CombinedRasterizer(rasterizer_list)
    return rasterizer
