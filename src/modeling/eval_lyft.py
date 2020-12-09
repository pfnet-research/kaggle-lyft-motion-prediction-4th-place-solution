import argparse
from distutils.util import strtobool
import numpy as np
import torch
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

# from l5kit.evaluation import write_pred_csv
from l5kit.data import LocalDataManager, ChunkedDataset
# from l5kit.dataset import AgentDataset

import sys
import os

from tqdm import tqdm

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.transforms.augmentation import _agent_type_onehot
from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch
from lib.dataset.faster_agent_dataset import FasterAgentDataset
from lib.evaluation.mask import load_mask_chopped
from modeling.load_flag import Flags
from lib.rasterization.rasterizer_builder import build_custom_rasterizer
from modeling.builder import build_multi_predictor, build_multi_mode_deep_ensemble
from lib.functions.transform import transform_points_batch
# from lib.utils.dotdict import DotDict
from lib.utils.yaml_utils import save_yaml, load_yaml
from src.lib.nn.models.single.lyft_model import LyftModel
from lib.nn.models.deep_ensemble.lyft_multi_deep_ensemble_predictor import LyftMultiDeepEnsemblePredictor
from lib.sampling.agent_sampling_changing_yaw import create_generate_agent_sample_changing_yaw_partial


# Referred https://www.kaggle.com/pestipeti/pytorch-baseline-inference
def run_prediction(predictor, data_loader,
                   convert_world_from_agent: bool = False,
                   feat_mode: str = "none"):
    predictor.eval()

    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []
    target_availabilities_list = []
    targets_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:
            image = data["image"].to(device)
            if feat_mode == "agent_type":
                x_feat = torch.tensor([
                    _agent_type_onehot(lp) for lp in data["label_probabilities"].cpu().numpy()
                ]).to(device)
                outputs = predictor(image, x_feat)
            else:
                outputs = predictor(image)

            if isinstance(predictor, LyftMultiDeepEnsemblePredictor):
                assert len(predictor.predictors) == len(outputs)
                pred = torch.cat([p for p, _ in outputs], dim=1)
                confidences = torch.cat([c for _, c in outputs], dim=1) / len(outputs)
            else:
                pred, confidences = outputs

            if convert_world_from_agent:
                # https://github.com/lyft/l5kit/blob/master/examples/agent_motion_prediction/agent_motion_prediction.ipynb
                # convert agent coordinates into world offsets
                agents_coords = pred  # (bs, num_modes, future_len, 2=xy)
                dtype = pred.dtype
                world_from_agents = data["world_from_agent"].type(dtype).to(device)  # (bs, 3, 3)
                centroids = data["centroid"].type(dtype).to(device)  # (bs, 2)

                bs, num_modes, future_len, cdim = agents_coords.shape
                agents_coords = agents_coords.reshape(bs * num_modes * future_len, cdim)
                transf_matrix = world_from_agents[:, None, None, :, :].expand(bs, num_modes, future_len, 3, 3).reshape(
                    bs * num_modes * future_len, 3, 3)
                centroids = centroids[:, :2]
                centroids = centroids[:, None, None, :].expand(bs, num_modes, future_len, 2).reshape(
                    bs * num_modes * future_len, 2)
                pred = transform_points_batch(agents_coords, transf_matrix) - centroids
                pred = pred.view(bs, num_modes, future_len, cdim)

            # errors = pytorch_neg_multi_log_likelihood_batch(
            #     torch.as_tensor(data["target_positions"], device=device),
            #     torch.as_tensor(pred, device=device),
            #     torch.as_tensor(confidences, device=device),
            #     torch.as_tensor(data["target_availabilities"], device=device),
            #     reduction="mean")
            # print("errors", errors)
            pred_coords_list.append(pred.cpu().numpy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps_list.append(data["timestamp"].numpy().copy())
            track_id_list.append(data["track_id"].numpy().copy())
            target_availabilities_list.append(data["target_availabilities"].numpy().copy())
            targets_list.append(data["target_positions"].numpy().copy())
    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    targets = np.concatenate(targets_list)
    target_availabilities = np.concatenate(target_availabilities_list)
    return timestamps, track_ids, coords, confs, targets, target_availabilities


def predict_and_save(predictor, valid_loader, convert_world_from_agent, out_dir, model_mode, debug, device,
                     feat_mode: str = "none"):
    # --- Inference ---
    (
        timestamps, track_ids, coords, confs, targets, target_availabilities
    ) = run_prediction(predictor, valid_loader, convert_world_from_agent, feat_mode=feat_mode)

    debug_str = "_debug" if debug else ""
    prediction_out_dir = out_dir / f"eval_{model_mode}{debug_str}"
    os.makedirs(str(prediction_out_dir), exist_ok=True)

    # --- Save to npz format, for future analysis purpose ---
    npz_path = prediction_out_dir / "pred.npz"
    np.savez_compressed(
        npz_path,
        timestamps=timestamps,
        track_ids=track_ids,
        coords=coords,
        confs=confs,
        targets=targets,
        target_availabilities=target_availabilities,
    )
    print(f"Saved to {npz_path}")

    # Evaluate loss
    errors = pytorch_neg_multi_log_likelihood_batch(
        torch.as_tensor(targets, device=device),
        torch.as_tensor(coords, device=device),
        torch.as_tensor(confs, device=device),
        torch.as_tensor(target_availabilities, device=device),
        reduction="none")
    print("errors", errors.shape, torch.mean(errors))
    # import IPython; IPython.embed()


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out', '-o', default='results/tmp',
                        help='Directory to output the result')
    parser.add_argument('--model_mode', type=str, default='ema',
                        help='')
    parser.add_argument('--yaw_delta', type=float, default=None, help="Use `yaw - yaw_delta`")
    # parser.add_argument('--convert_world_from_agent', '-c', type=strtobool, default='false',
    #                     help='Convert agent coord to world or not. Should be False for calculating loss')
    parser.add_argument('--debug', '-d', type=strtobool, default='false',
                        help='')
    parser.add_argument('--n_valid_data', '-v', type=int, default=-1,
                        help='Number of valid data, -1 is to use all data')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Number of valid data, -1 is to use all data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    out_dir = Path(args.out)
    debug = args.debug

    flags_dict = load_yaml(out_dir / 'flags.yaml')
    cfg = load_yaml(out_dir / 'cfg.yaml')
    # flags = DotDict(flags_dict)
    flags = Flags()
    flags.update(flags_dict)
    print(f"flags: {flags_dict}")

    # set env variable for data
    # Not use flags.l5kit_data_folder, but use fixed test data.
    l5kit_data_folder = "../../input/lyft-motion-prediction-autonomous-vehicles"
    os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
    dm = LocalDataManager(None)

    print("Load dataset...")
    # default_test_cfg = {
    #     'key': 'scenes/test.zarr',
    #     'batch_size': 32,
    #     'shuffle': False,
    #     'num_workers': 4
    # }
    # test_cfg = cfg.get("test_data_loader", default_test_cfg)

    # Rasterizer
    rasterizer = build_custom_rasterizer(cfg, dm, eval=True)

    valid_cfg = cfg["valid_data_loader"]
    # valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]
    valid_path = valid_cfg["key"]
    valid_agents_mask = None
    if flags.validation_chopped:
        num_frames_to_chop = 100
        th_agent_prob = cfg["raster_params"]["filter_agents_threshold"]
        min_frame_future = 1
        num_frames_to_copy = num_frames_to_chop
        valid_agents_mask = load_mask_chopped(
            dm.require(valid_path), th_agent_prob, num_frames_to_copy, min_frame_future)
        print("valid_path", valid_path, "valid_agents_mask", valid_agents_mask.shape)
    valid_zarr = ChunkedDataset(dm.require(valid_path)).open(cached=False)
    valid_agent_dataset = FasterAgentDataset(
        cfg, valid_zarr, rasterizer, agents_mask=valid_agents_mask,
        min_frame_history=flags.min_frame_history, min_frame_future=flags.min_frame_future,
        override_sample_function_name=flags.override_sample_function_name,
    )

    if args.yaw_delta is not None:
        assert flags.override_sample_function_name == ""
        valid_agent_dataset.ego_dataset.sample_function = create_generate_agent_sample_changing_yaw_partial(
            cfg, rasterizer, args.yaw_delta
        )

    # valid_dataset = TransformDataset(valid_agent_dataset, transform)
    valid_dataset = valid_agent_dataset

    # Only use `n_valid_data` dataset for fast check.
    # Sample dataset from regular interval, to increase variety/coverage
    # n_valid_data = 1000 if debug else flags.n_valid_data
    n_valid_data = 1000 if debug else args.n_valid_data
    print("n_valid_data", n_valid_data)
    if n_valid_data > 0 and n_valid_data < len(valid_dataset):
        valid_sub_indices = np.linspace(0, len(valid_dataset)-1, num=n_valid_data, dtype=np.int64)
        valid_dataset = Subset(valid_dataset, valid_sub_indices)
    valid_batchsize = valid_cfg["batch_size"]

    local_valid_dataset = valid_dataset
    collate_fn = None
    valid_loader = DataLoader(
        local_valid_dataset, valid_batchsize, shuffle=False,
        pin_memory=True, num_workers=valid_cfg["num_workers"],
        collate_fn=collate_fn)

    print(valid_agent_dataset)
    print("# AgentDataset test:", len(valid_agent_dataset))
    print("# ActualDataset test:", len(valid_dataset))
    in_channels, height, width = valid_agent_dataset[0]["image"].shape  # get input image shape
    print("in_channels", in_channels, "height", height, "width", width)

    # ==== INIT MODEL
    # device = torch.device(flags.device)
    device = torch.device(args.device)

    if flags.pred_mode == "single":
        predictor = LyftModel(cfg)
    elif flags.pred_mode == "multi":
        predictor = build_multi_predictor(cfg, flags, device, in_channels)
    elif flags.pred_mode == "multi_deep_ensemble":
        predictor = build_multi_mode_deep_ensemble(cfg, flags, device, in_channels)
    else:
        raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

    model_mode = args.model_mode
    debug_str = "_debug" if debug else ""

    if model_mode == "original":
        pt_path = out_dir/"predictor.pt"
    elif model_mode == "ema":
        pt_path = out_dir/"predictor_ema.pt"
    elif model_mode == "cycle0":
        pt_path = out_dir/"snapshot_0th_cycle.pt"
    else:
        raise ValueError(f"[ERROR] Unexpected value model_mode={model_mode}")
    print(f"Loading from {pt_path}")
    try:
        predictor.load_state_dict(torch.load(str(pt_path)))
    except RuntimeError:
        print("Load from predictor failed, loading from predictor.base_model...")
        predictor.base_model.load_state_dict(torch.load(str(pt_path)))
    # Use this instead for old code, before MultiPredictor refactoring.
    # predictor.base_model.load_state_dict(torch.load(str(pt_path)))
    predictor.to(device)

    convert_world_from_agent = False

    if args.yaw_delta is not None:
        save_dir = out_dir / str(args.yaw_delta)
    else:
        save_dir = out_dir

    if isinstance(predictor, LyftMultiDeepEnsemblePredictor):
        for k, name in enumerate(predictor.names):
            print(f"Predicting {name}...")
            predict_and_save(
                predictor.get_kth_predictor(k),
                valid_loader,
                convert_world_from_agent,
                save_dir / name,
                model_mode,
                debug,
                device,
                flags.feat_mode
            )
        pass
    else:
        predict_and_save(predictor, valid_loader, convert_world_from_agent, save_dir, model_mode, debug, device,
                         flags.feat_mode)
