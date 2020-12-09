import dataclasses
import numpy as np
import os
import torch
from pathlib import Path

from pytorch_pfn_extras.training import IgniteExtensionsManager

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, ConcatDataset
import pytorch_pfn_extras.training.extensions as E
from ignite.engine import Events

from l5kit.data import LocalDataManager, ChunkedDataset
import sys

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.nn.models.yaw.lyft_yaw_regressor import LyftYawRegressor
from lib.nn.models.yaw.yaw_predictor import LyftYawPredictor
from lib.dataset.multi_agent_dataset import MultiAgentDataset
from lib.rasterization.rasterizer_builder import build_custom_rasterizer
from lib.evaluation.mask import load_mask_chopped
from lib.training.exponential_moving_average import EMA
from lib.dataset.faster_agent_dataset import FasterAgentDataset
from lib.training.distributed_evaluator import DistributedEvaluator
from lib.functions.nll import pytorch_neg_multi_log_likelihood_single
from lib.nn.models.single.lyft_regressor import LyftRegressor
from lib.nn.models.multi.lyft_multi_regressor import LyftMultiRegressor
from lib.nn.models.rnn_head_multi.rnn_head_multi_regressor import RNNHeadMultiRegressor
from lib.nn.models.deep_ensemble.lyft_multi_deep_ensemble_regressor import LyftMultiDeepEnsembleRegressor
from lib.dataset.transform_dataset import TransformDataset
from lib.training.ignite_utils import create_trainer
from lib.utils.distributed_utils import setup_distributed, split_valid_dataset
from src.lib.nn.models.single.lyft_model import LyftModel
from lib.utils.yaml_utils import save_yaml, load_yaml
from lib.transforms import pred_mode_to_transform, pred_mode_to_collate_fn
from lib.nn.models.multi_agent.lyft_multi_agent_regressor import LyftMultiAgentRegressor
from modeling.load_flag import load_flags, Flags
from modeling.builder import (
    build_multi_agent_predictor,
    build_multi_mode_deep_ensemble,
    build_multi_predictor,
    build_rnn_head_multi_predictor,
)
from lib.training.scene_sampler import SceneSampler, DistributedSceneSampler
from lib.transforms.augmentation import ImageAugmentation
from lib.training.lr_scheduler import LRScheduler
from lib.training.snapshot_object_when_lr_increase import SnapshotObjectWhenLRIncrease
from lib.utils.resumable_distributed_sampler import ResumableDistributedSampler
from lib.functions.nll import pytorch_weighted_neg_multi_log_likelihood_batch, pytorch_neg_multi_log_likelihood_batch


if __name__ == '__main__':
    # --- Distributed initial setup ---
    is_mpi, rank, world_size, local_rank = setup_distributed()
    print(f"is_mpi {is_mpi}, rank {rank}, world_size {world_size}, local_rank {local_rank}")
    # --- Distributed initial setup done ---

    mode = "distributed" if is_mpi else ""
    flags: Flags = load_flags(mode=mode)
    flags_dict = dataclasses.asdict(flags)
    cfg = load_yaml(flags.cfg_filepath)
    out_dir = Path(flags.out_dir)
    if local_rank == 0:
        print(f"cfg {cfg}")
        os.makedirs(str(out_dir), exist_ok=True)
        print(f"flags: {flags_dict}")
        save_yaml(out_dir / 'flags.yaml', flags_dict)
        save_yaml(out_dir / 'cfg.yaml', cfg)
    debug = flags.debug

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder
    dm = LocalDataManager(None)

    # ===== INIT DATASET
    if local_rank == 0:
        print("INIT DATASET...")
    train_cfg = cfg["train_data_loader"]
    valid_cfg = cfg["valid_data_loader"]

    # Rasterizer
    rasterizer = build_custom_rasterizer(cfg, dm)
    rasterizer_eval = build_custom_rasterizer(cfg, dm, eval=True)
    print("rasterizer", rasterizer)

    # Train dataset/dataloader
    if flags.pred_mode in ["multi_agent", "rnn_head_multi", "yaw"]:
        transform = pred_mode_to_transform[flags.pred_mode]
    else:
        transform = ImageAugmentation(flags).transform
    transform_validation = transform
    if not flags.augmentation_in_validation:
        transform_validation = pred_mode_to_transform[flags.pred_mode]

    collate_fn = pred_mode_to_collate_fn[flags.pred_mode]

    train_path = "scenes/sample.zarr" if debug else train_cfg["key"]
    train_zarr = ChunkedDataset(dm.require(train_path)).open(cached=False)
    if local_rank == 0:
        print("train_zarr", type(train_zarr))
        print(f"Open Dataset {flags.pred_mode}...")
    # train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)

    valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]
    valid_zarr = ChunkedDataset(dm.require(valid_path)).open(cached=False)

    if flags.pred_mode == "multi_agent":
        train_agent_dataset = MultiAgentDataset(
            cfg, train_zarr, rasterizer, min_frame_history=flags.min_frame_history,
            min_frame_future=flags.min_frame_future
        )
        if flags.include_valid:
            train_agent_dataset2 = MultiAgentDataset(
                cfg, valid_zarr, rasterizer, min_frame_history=flags.min_frame_history,
                min_frame_future=flags.min_frame_future
            )
        print("Before Concat: ", len(train_agent_dataset), len(train_agent_dataset2))
        train_agent_dataset = ConcatDataset([train_agent_dataset, train_agent_dataset2])
        print("After  Concat: ", len(train_agent_dataset))
    else:
        train_agent_dataset = FasterAgentDataset(
            cfg, train_zarr, rasterizer, min_frame_history=flags.min_frame_history,
            min_frame_future=flags.min_frame_future,
            override_sample_function_name=flags.override_sample_function_name,
        )
        if flags.include_valid:
            train_agent_dataset2 = FasterAgentDataset(
                cfg, valid_zarr, rasterizer, min_frame_history=flags.min_frame_history,
                min_frame_future=flags.min_frame_future,
                override_sample_function_name=flags.override_sample_function_name,
            )
            print("Before Concat: ", len(train_agent_dataset), len(train_agent_dataset2))
            train_agent_dataset = ConcatDataset([train_agent_dataset, train_agent_dataset2])
            print("After  Concat: ", len(train_agent_dataset))

    train_dataset = TransformDataset(train_agent_dataset, transform)
    if debug:
        # Only use 1000 dataset for fast check...
        train_dataset = Subset(train_dataset, np.arange(1000))
    if is_mpi:
        if flags.scene_sampler:
            assert isinstance(train_agent_dataset, FasterAgentDataset)
            train_sampler = DistributedSceneSampler(
                get_frame_arguments=train_agent_dataset.get_frame_arguments,
                min_state_index=flags.scene_sampler_min_state_index,
                shuffle=bool(train_cfg["shuffle"])
            )
        else:
            train_sampler = ResumableDistributedSampler(train_dataset, shuffle=bool(train_cfg["shuffle"]))
    else:
        if flags.scene_sampler:
            assert isinstance(train_agent_dataset, FasterAgentDataset)
            assert bool(train_cfg["shuffle"])
            train_sampler = SceneSampler(
                get_frame_arguments=train_agent_dataset.get_frame_arguments,
                min_state_index=flags.scene_sampler_min_state_index,
            )
        else:
            train_sampler = None
    shuffle = bool(train_cfg["shuffle"]) if train_sampler is None else False  # Cannot use shuffle when sampler is set.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        # num_workers=0,
        shuffle=shuffle,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    if local_rank == 0:
        print("train_agent_dataset", len(train_agent_dataset))
        print(train_agent_dataset)

    valid_agents_mask = None
    if flags.validation_chopped:
        num_frames_to_chop = 100
        th_agent_prob = cfg["raster_params"]["filter_agents_threshold"]
        min_frame_future = 1
        num_frames_to_copy = num_frames_to_chop
        valid_agents_mask = load_mask_chopped(
            dm.require(valid_path), th_agent_prob, num_frames_to_copy, min_frame_future)
        print("valid_path", valid_path, "valid_agents_mask", valid_agents_mask.shape)

    if local_rank == 0:
        print("valid_zarr", type(valid_zarr))
    # valid_agent_dataset = AgentDataset(cfg, valid_zarr, rasterizer, agents_mask=valid_agents_mask)
    if flags.pred_mode == "multi_agent":
        if valid_agents_mask is not None:
            print("agents_mask for MultiAgentDataset is not supported and simply ignored now!")
        valid_agent_dataset = MultiAgentDataset(
            cfg, valid_zarr, rasterizer_eval,
            min_frame_history=flags.min_frame_history, min_frame_future=flags.min_frame_future
        )
    else:
        valid_agent_dataset = FasterAgentDataset(
            cfg, valid_zarr, rasterizer_eval, agents_mask=valid_agents_mask,
            min_frame_history=flags.min_frame_history, min_frame_future=flags.min_frame_future,
            override_sample_function_name=flags.override_sample_function_name,
        )
    valid_dataset = TransformDataset(valid_agent_dataset, transform_validation)

    # Only use `n_valid_data` dataset for fast check.
    # Sample dataset from regular interval, to increase variety/coverage
    n_valid_data = flags.n_valid_data
    if n_valid_data < len(valid_dataset):
        valid_sub_indices = np.linspace(0, len(valid_dataset) - 1, num=n_valid_data, dtype=np.int64)
        valid_dataset = Subset(valid_dataset, valid_sub_indices)

    valid_batchsize = valid_cfg["batch_size"]

    if is_mpi:
        local_valid_dataset = split_valid_dataset(valid_dataset, rank, world_size)
    else:
        local_valid_dataset = valid_dataset
    valid_loader = DataLoader(local_valid_dataset, valid_batchsize, shuffle=False,
                              pin_memory=True, num_workers=valid_cfg["num_workers"],
                              collate_fn=collate_fn)

    if local_rank == 0:
        print(valid_agent_dataset)
        print("# AgentDataset train:", len(train_agent_dataset), "#valid", len(valid_agent_dataset))
        print("# ActualDataset train:", len(train_dataset), "#valid", len(valid_dataset))
        # AgentDataset train: 22496709 #valid 21624612
        # ActualDataset train: 100 #valid 100

    # ==== INIT MODEL
    in_channels, height, width = train_dataset[0][0].shape  # get input image shape
    print("in_channels", in_channels, "height", height, "width", width)
    device = torch.device(f"cuda:{local_rank}") if is_mpi else torch.device(flags.device)

    if flags.target_scale_filepath:
        target_scale = torch.as_tensor(np.load(flags.target_scale_filepath)["target_scale"])
        if local_rank == 0:
            print("target_scale", target_scale)
    else:
        target_scale = None

    if flags.pred_mode == "single":
        predictor = LyftModel(cfg)
        regressor = LyftRegressor(predictor, lossfun=pytorch_neg_multi_log_likelihood_single)
    elif flags.pred_mode == "multi":
        predictor = build_multi_predictor(cfg, flags, device, in_channels, target_scale=target_scale)
        lossfun = {
            "pytorch_neg_multi_log_likelihood_batch": pytorch_neg_multi_log_likelihood_batch,
            "pytorch_weighted_neg_multi_log_likelihood_batch": pytorch_weighted_neg_multi_log_likelihood_batch
        }[flags.lossfun]
        regressor = LyftMultiRegressor(predictor, lossfun=lossfun)
    elif flags.pred_mode == "multi_agent":
        predictor = build_multi_agent_predictor(cfg, flags, device, in_channels)
        regressor = LyftMultiAgentRegressor(predictor)
    elif flags.pred_mode == "rnn_head_multi":
        predictor = build_rnn_head_multi_predictor(cfg, flags, device, in_channels, target_scale=target_scale)
        regressor = RNNHeadMultiRegressor(predictor)
    elif flags.pred_mode == "multi_deep_ensemble":
        predictor = build_multi_mode_deep_ensemble(cfg, flags, device, in_channels, target_scale=target_scale)
        regressor = LyftMultiDeepEnsembleRegressor(predictor)
    elif flags.pred_mode == "yaw":
        multi_predictor = build_multi_predictor(cfg, flags, device, in_channels, target_scale=target_scale)
        predictor = LyftYawPredictor(multi_predictor.base_model)
        regressor = LyftYawRegressor(predictor, lossfun=flags.lossfun)
    else:
        raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

    if flags.load_predictor_filepath:
        if local_rank == 0:
            print(f"Loading from {flags.load_predictor_filepath}")
        predictor.load_state_dict(torch.load(flags.load_predictor_filepath, map_location=device))

    regressor.to(device)
    if is_mpi:
        model = nn.parallel.DistributedDataParallel(
            regressor, device_ids=[local_rank], find_unused_parameters=True
        )
    else:
        model = regressor
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train setup
    trainer = create_trainer(model, optimizer, device)

    # NOTE: Load predictor_ema.pt once to resume the ``shadow`` of EMA.
    # The resume of the predictor or optimizer is done later by the autoload function of ``E.snapshot``.
    predictor_ema_filepath = (out_dir / "predictor_ema.pt")
    if flags.resume_if_possible and predictor_ema_filepath.exists():
        if local_rank == 0:
            print(f"Resume: Loading from {predictor_ema_filepath}")
        predictor.load_state_dict(torch.load(str(predictor_ema_filepath), map_location=device))
    ema = EMA(predictor, decay=flags.ema_decay)

    def eval_func(*batch):
        loss, metrics = model(*[elem.to(device) for elem in batch])
        # HACKING: report ema value with prefix.
        if flags.ema_decay > 0:
            regressor.prefix = "ema_"
            ema.assign()
            loss, metrics = model(*[elem.to(device) for elem in batch])
            ema.resume()
            regressor.prefix = ""

    if is_mpi:
        valid_evaluator = DistributedEvaluator(
            valid_loader,
            model,
            progress_bar=True,
            eval_func=eval_func,
            local_rank=local_rank,
            world_size=world_size,
            device=device
        )
    else:
        valid_evaluator = E.Evaluator(
            valid_loader,
            model,
            progress_bar=True,
            eval_func=eval_func,
            device=device
        )

    log_trigger = (10 if debug else 1000, "iteration")
    log_report = E.LogReport(
        trigger=log_trigger, filename=f"log_rank{local_rank}"
    )
    extensions = [
        log_report,
    ]

    if local_rank == 0:
        extensions.extend([
            # E.LogReport(trigger=(1, "epoch")),
            # log_report,  # Save `log` to file
            E.ProgressBar(update_interval=10 if debug else 100),  # Show progress bar during training
            E.PrintReport(),  # Print "log" to terminal
            # E.FailOnNonNumber()  # Stop training when nan is detected.
        ])

    # batch_size = train_cfg["batch_size"]
    # max_num_steps = cfg["train_params"]["max_num_steps"]
    # epoch = ceil(batch_size * max_num_steps / len(train_dataset))
    # print(f"Target step {max_num_steps} -> epoch {epoch}")
    epoch = flags.epoch

    models = {"main": model}
    optimizers = {"main": optimizer}
    manager = IgniteExtensionsManager(
        trainer,
        models,
        optimizers,
        epoch,
        extensions=extensions,
        out_dir=str(out_dir),
    )
    # Run evaluation for valid dataset in each epoch.
    manager.extend(valid_evaluator, trigger=(flags.validation_freq, "iteration"))
    if local_rank == 0:
        # Save predictor.pt every epoch
        manager.extend(E.snapshot_object(predictor, "predictor.pt"),
                       trigger=(flags.snapshot_freq, "iteration"))
        # Check & Save best validation predictor.pt every epoch
        # manager.extend(E.snapshot_object(predictor, "best_predictor.pt"),
        #                trigger=MinValueTrigger("validation/mainmodule/nll",
        #                trigger=(flags.snapshot_freq, "iteration")))

    # --- lr scheduler ---
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)
    if flags.scheduler_type != "":
        scheduler_type = flags.scheduler_type

        # For backward compatibility
        if scheduler_type == "exponential":
            scheduler_type = "ExponentialLR"

        manager.extend(
            LRScheduler(optimizer, scheduler_type, flags.scheduler_kwargs), trigger=flags.scheduler_trigger
        )

    manager.extend(E.observe_lr(optimizer=optimizer), trigger=log_trigger)

    if flags.ema_decay > 0:
        # Exponential moving average
        manager.extend(lambda manager: ema(), trigger=(1, "iteration"))
        if local_rank == 0:
            def save_ema_model(manager):
                ema.assign()
                torch.save(predictor.state_dict(), out_dir / "predictor_ema.pt")
                ema.resume()
            manager.extend(save_ema_model, trigger=(flags.snapshot_freq, "iteration"))

    if is_mpi and (train_sampler is not None) and hasattr(train_sampler, "set_epoch"):
        manager.extend(lambda manager: train_sampler.set_epoch(manager.epoch), trigger=(1, "epoch"))

    saver_rank = 0 if is_mpi else None
    manager.extend(SnapshotObjectWhenLRIncrease(predictor, optimizer, saver_rank=saver_rank))
    manager.extend(
        E.snapshot(n_retains=1, autoload=flags.resume_if_possible, saver_rank=saver_rank),
        trigger=(flags.snapshot_freq, "iteration")
    )

    if (train_sampler is not None) and hasattr(train_sampler, "resume"):
        @trainer.on(Events.STARTED)
        def resume_train_sampler(engine):
            print("resume_train_sampler:", engine.state.iteration, engine.state.epoch)
            train_sampler.resume(engine.state.iteration, engine.state.epoch)

    trainer.run(train_loader, max_epochs=epoch)

    if local_rank == 0:
        torch.save(predictor.state_dict(), out_dir / 'predictor_last.pt')
        df = log_report.to_dataframe()
        df.to_csv(out_dir / "log.csv", index=False)
