import sys
import os
from typing import Optional
from collections import defaultdict
import re

import pretrainedmodels
import torch
from torch import nn
import timm

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.nn.models.multi.multi_model_predictor import LyftMultiModelPredictor
from lib.nn.models.multi.pretrained_cnn_multi import PretrainedCNNMulti
from lib.nn.models.multi.resnest_multi import ResNeStMulti
from lib.nn.models.multi.lyft_multi_model import LyftMultiModel
from lib.nn.models.multi.efficientnet_multi import EfficientNetMulti
try:
    from lib.nn.models.multi.timm_multi import TimmMulti
except Exception as e:
    print("[WARNING] TimmMulti import failed!")
from lib.nn.models.deep_ensemble.lyft_multi_deep_ensemble_predictor import LyftMultiDeepEnsemblePredictor
from lib.nn.models.rnn_head_multi.lstm_head_multi_predictor import LSTMHeadMultiPredictor
from lib.nn.models.rnn_head_multi.target_scale_wrapper import TargetScaleWrapper
from modeling.load_flag import Flags


def build_rnn_head_multi_predictor(
    cfg, flags: Flags, device: torch.device, in_channels: int, target_scale: Optional[torch.Tensor] = None
) -> nn.Module:
    num_modes = 3
    future_len = cfg["model_params"]["future_num_frames"]
    model_name = flags.model_name
    print("model_name", model_name, "model_kwargs", flags.model_kwargs)

    if model_name == "LSTMHeadMultiPredictor":
        predictor = LSTMHeadMultiPredictor(
            num_modes=num_modes, future_len=future_len, in_channels=in_channels, **flags.model_kwargs
        )
    else:
        raise ValueError(f"[ERROR] Unexpected value model_name={model_name}")

    if target_scale is not None:
        assert target_scale.shape == (future_len, 2)
        predictor = TargetScaleWrapper(predictor, target_scale)

    # --- Forward once to initialize lazy params ---
    bs = 2
    height, width = cfg["raster_params"]["raster_size"]
    in_channels = predictor.in_channels
    image = torch.rand((bs, in_channels, height, width), dtype=torch.float32).to(device)
    history_positions = torch.rand((bs, 10, 2), dtype=torch.float32).to(device)
    history_availablelities = torch.ones((bs, 10), dtype=torch.float32).to(device)
    predictor.to(device)
    pred, confidences = predictor(image, history_positions, history_availablelities)
    assert pred.shape == (bs, num_modes, future_len, 2)
    assert confidences.shape == (bs, num_modes)
    # --- Done ---

    return predictor


def build_multi_predictor(
    cfg,
    flags: Flags,
    device: torch.device,
    in_channels: int,
    target_scale: Optional[torch.Tensor] = None,
    num_modes: int = 3,
) -> nn.Module:
    model_name = flags.model_name
    print("model_name", model_name, "model_kwargs", flags.model_kwargs, "num_modes", num_modes)
    if model_name == "resnet18":
        print("Building LyftMultiModel")
        base_model = LyftMultiModel(cfg, num_modes=num_modes, in_channels=in_channels)
    elif "efficientnet" in model_name:
        print("Building EfficientNetMulti")
        base_model = EfficientNetMulti(
            cfg, num_modes=num_modes, model_name=model_name, in_channels=in_channels, **flags.model_kwargs)
    elif "resnest" in model_name:
        print("Building ResNeStMulti")
        base_model = ResNeStMulti(
            cfg, num_modes=num_modes, model_name=model_name, in_channels=in_channels, **flags.model_kwargs)
    elif model_name in pretrainedmodels.__dict__.keys():
        print("Building PretrainedCNNMulti")
        base_model = PretrainedCNNMulti(
            cfg, num_modes=num_modes, model_name=model_name, in_channels=in_channels, **flags.model_kwargs)
    elif model_name in timm.list_models():
        print("Building TimmMulti")
        base_model = TimmMulti(
            cfg, in_channels=in_channels, num_modes=num_modes, backbone=model_name, **flags.model_kwargs
        )
    else:
        raise ValueError(f"[ERROR] Unexpected value model_name={model_name}")
    predictor = LyftMultiModelPredictor(base_model, cfg, num_modes=num_modes, target_scale=target_scale)

    # --- Forward once to initialize lazy params ---
    bs = 2
    height, width = cfg["raster_params"]["raster_size"]
    in_channels = predictor.in_channels
    x = torch.rand((bs, in_channels, height, width), dtype=torch.float32).to(device)
    predictor.to(device)
    if flags.feat_mode == "agent_type":
        feat_channels = flags.model_kwargs["feat_channels"]
        x_feat = torch.rand((bs, feat_channels), dtype=torch.float32).to(device)
        predictor(x, x_feat)
    else:
        predictor(x)
    # --- Done ---

    return predictor


def build_multi_mode_deep_ensemble(
    cfg,
    flags: Flags,
    device: torch.device,
    in_channels: int,
    target_scale: Optional[torch.Tensor] = None,
) -> nn.Module:
    ensemble_name = flags.model_name
    model_names = ensemble_name.split("+")
    use_D4 = flags.model_kwargs.pop("use_D4", False)
    print("use_D4:", use_D4)

    predictors = []
    names = []
    cnts = defaultdict(int)
    for model_name in model_names:
        match_obj = re.match(r"(.+)_(\d+)modes", model_name)
        if match_obj is None:
            actual_model_name = model_name
            num_modes = 3
        else:
            actual_model_name = match_obj.group(1)
            num_modes = int(match_obj.group(2))

        flags.model_name = actual_model_name
        predictor = build_multi_predictor(cfg, flags, device, in_channels, target_scale, num_modes)
        flags.model_name = ensemble_name

        predictors.append(predictor)
        names.append(f"{model_name}_{cnts[model_name]}")
        cnts[model_name] += 1

    predictor = LyftMultiDeepEnsemblePredictor(predictors, names, use_D4)
    flags.model_kwargs["use_D4"] = use_D4
    return predictor


def build_multi_agent_predictor(cfg, flags: Flags, device: torch.device, in_channels: int) -> nn.Module:
    num_modes = 3
    model_name = flags.model_name
    print("model_name", model_name, "model_kwargs", flags.model_kwargs)
    # TODO: now model_name is skipped...
    print("Building LyftMultiModel")
    if model_name.startswith("smp_"):
        from lib.nn.models.multi_agent.smp_multi_agent_model import SMPMultiAgentModel
        predictor = SMPMultiAgentModel(cfg, num_modes=num_modes, in_channels=in_channels, **flags.model_kwargs)
    else:
        raise NotImplementedError(f"model_name {model_name} is not supported for multi_agent model")

    # TODO:
    # --- Forward once to initialize lazy params ---
    bs = 2
    n_agents = 2
    height, width = cfg["raster_params"]["raster_size"]
    in_channels = predictor.in_channels
    x = torch.rand((bs, in_channels, height, width), dtype=torch.float32).to(device)
    centroid_pixel = torch.randint(3, 10, (n_agents, 2), dtype=torch.long).to(device)
    batch_agents = torch.tensor([0, 0], dtype=torch.long).to(device)
    predictor.to(device)
    predictor(x, centroid_pixel, batch_agents)
    # --- Done ---
    return predictor
