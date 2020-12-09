import argparse
from copy import deepcopy
from distutils.util import strtobool

from typing import Dict, Any, Tuple

from dataclasses import dataclass, field

from lib.utils.yaml_utils import load_yaml


@dataclass
class Flags:
    # --- Overall configs ---
    debug: bool = False
    cfg_filepath: str = "configs/0905_cfg.yaml"
    # --- Data configs ---
    l5kit_data_folder: str = "../../input/lyft-motion-prediction-autonomous-vehicles"
    min_frame_history: int = 10  # minimum frame history used in AgentDataset
    min_frame_future: int = 1  # minimum frame future used in AgentDataset
    override_sample_function_name: str = ""  # override sample function. Ex. "generate_agent_sample_tl_history"
    # --- Model configs ---
    pred_mode: str = "multi"
    model_name: str = "resnet18"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    target_scale_filepath: str = ""
    # --- Training configs ---
    device: str = "cuda:0"
    out_dir: str = "results/multi_train"
    epoch: int = 2
    snapshot_freq: int = 5000
    scheduler_type: str = "exponential"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {"gamma": 0.999999})
    scheduler_trigger: Tuple[int, str] = (1, "iteration")
    ema_decay: float = 0.999  # negative value is to inactivate ema.
    validation_freq: int = 20000  # validation frequency
    validation_chopped: bool = False  # use chopped validation dataset or not.
    n_valid_data: int = 10000  # number of validation data
    resume_if_possible: bool = True  # Resume when predictor.pt is found in outdir
    load_predictor_filepath: str = ""  # Start from this pretrained predictor, if specified
    scene_sampler: bool = False  # Generate one example per scene, if specified
    scene_sampler_min_state_index: int = 0  # min_state_index for SceneSampler
    augmentation_in_validation : bool = False # apply augmentation in validation
    cutout: Dict[str, Any] = field(default_factory=lambda: {"p": 0.0, "scale_min": 0.75, "scale_max": 0.99})  # "p": 0 means no augmentation
    blur: Dict[str, Any] = field(default_factory=lambda: {"p": 0.0, "blur_limit": [3, 5]})  # "p": 0 means no augmentation
    downscale: Dict[str, Any] = field(default_factory=lambda: {"p": 0.0, "num_holes": 5, "max_h_size": 20, "max_w_size": 20, "fill_value": 0})  # "p": 0 means no augmentation
    crossdrop: Dict[str, Any] = field(default_factory=lambda: {"p": 0.0, "max_h_cut": 0.3, "max_w_cut": 0.3, "fill_value": 0})  # "p": 0 means no augmentation
    feat_mode: str = "none"  # Append `x_feat` feature. "agent_type" is supported now.
    include_valid: bool = False  # Include validation dataset as train data or not.
    lossfun: str = "pytorch_neg_multi_log_likelihood_batch"  # Loss function

    def update(self, param_dict: Dict):
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)


def load_flags(mode="") -> Flags:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--yaml_filepath', '-y', type=str, default="./flags/20200905_sample_flag.yaml",
                        help='Flags yaml file path')
    args = parser.parse_args()

    # --- Default setting ---

    flags = load_yaml(args.yaml_filepath)
    # print("yaml flags", flags)
    base_flags = Flags()
    base_flags.update(flags)
    return base_flags
