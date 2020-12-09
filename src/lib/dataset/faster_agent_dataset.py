import bisect
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from l5kit.data import ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer
from l5kit.dataset.agent import MIN_FRAME_HISTORY, MIN_FRAME_FUTURE
from numcodecs import Blosc
from torch.utils.data import Dataset
from tqdm import tqdm
import zarr

from lib.dataset.custom_ego_dataset import get_frame_custom
from lib.dataset.fast_agent_dataset import FastAgentDataset
from lib.sampling.agent_sampling_tl_history import create_generate_agent_sample_tl_history_partial
from lib.sampling.agent_sampling_fixing_yaw import create_generate_agent_sample_fixing_yaw_partial


def _job(index):
    global agent_dataset
    track_id = agent_dataset.dataset.agents[index]["track_id"]
    frame_index = bisect.bisect_right(agent_dataset.cumulative_sizes_agents, index)
    scene_index = bisect.bisect_right(agent_dataset.cumulative_sizes, frame_index)

    if scene_index == 0:
        state_index = frame_index
    else:
        state_index = frame_index - agent_dataset.cumulative_sizes[scene_index - 1]

    return track_id, scene_index, state_index


class FasterAgentDataset(Dataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
        agents_mask: Optional[np.ndarray] = None,
        min_frame_history: int = MIN_FRAME_HISTORY,
        min_frame_future: int = MIN_FRAME_FUTURE,
        override_sample_function_name: str = "",
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"
        self.cfg = cfg
        self.ego_dataset = EgoDataset(cfg, zarr_dataset, rasterizer, perturbation)
        self.get_frame_arguments = self.load_get_frame_arguments(agents_mask, min_frame_history, min_frame_future)

        if override_sample_function_name != "":
            print("override_sample_function_name", override_sample_function_name)
        if override_sample_function_name == "generate_agent_sample_tl_history":
            self.ego_dataset.sample_function = create_generate_agent_sample_tl_history_partial(cfg, rasterizer)
        elif override_sample_function_name == "generate_agent_sample_fixing_yaw":
            self.ego_dataset.sample_function = create_generate_agent_sample_fixing_yaw_partial(cfg, rasterizer)

    def load_get_frame_arguments(
        self,
        agents_mask: Optional[np.ndarray] = None,
        min_frame_history: int = MIN_FRAME_HISTORY,
        min_frame_future: int = MIN_FRAME_FUTURE,
    ) -> zarr.core.Array:
        """
        Returns:
            zarr.core.Array: int64 array of (track_id, scene_index, state_index)
        """
        agent_prob = self.cfg["raster_params"]["filter_agents_threshold"]
        agents_mask_str = "" if agents_mask is None else f"_mask{np.sum(agents_mask)}"
        filename = f"get_frame_arguments_{agent_prob}_{min_frame_history}_{min_frame_future}{agents_mask_str}.zarr"
        cache_path = Path(self.ego_dataset.dataset.path) / filename

        if not cache_path.exists():
            global agent_dataset
            print(f"Cache {cache_path} does not exist, creating...")

            # Use FastAgentDataset to build agent_indices.
            agent_dataset = FastAgentDataset(
                cfg=self.cfg,
                zarr_dataset=self.ego_dataset.dataset,
                rasterizer=self.ego_dataset.rasterizer,
                perturbation=self.ego_dataset.perturbation,
                agents_mask=agents_mask,
                min_frame_history=min_frame_history,
                min_frame_future=min_frame_future
            )

            indices = agent_dataset.agents_indices
            with ProcessPoolExecutor(max_workers=16) as executor:
                get_frame_arguments = list(tqdm(executor.map(_job, indices, chunksize=10000), total=len(indices)))

            del agent_dataset

            get_frame_arguments = np.asarray(get_frame_arguments, dtype=np.int64)
            compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE, blocksize=0)
            z = zarr.open(str(cache_path), mode="w", shape=get_frame_arguments.shape, chunks=(20000, 3), dtype="i8", compressor=compressor)
            z[:] = get_frame_arguments

        z = zarr.open(str(cache_path), mode="r")
        return z

    def __len__(self) -> int:
        return len(self.get_frame_arguments)

    def __getitem__(self, index: int) -> dict:
        track_id, scene_index, state_index = self.get_frame_arguments[index]
        # return self.ego_dataset.get_frame(scene_index, state_index, track_id=track_id)
        return get_frame_custom(self.ego_dataset, scene_index, state_index, track_id=track_id)


if __name__ == "__main__":
    from l5kit.data import LocalDataManager
    from l5kit.rasterization import build_rasterizer
    from lib.utils.yaml_utils import load_yaml

    repo_root = Path(__file__).parent.parent.parent.parent

    dm = LocalDataManager(local_data_folder=str(repo_root / "input" / "lyft-motion-prediction-autonomous-vehicles"))
    dataset = ChunkedDataset(dm.require("scenes/sample.zarr")).open(cached=False)
    cfg = load_yaml(repo_root / "src" / "modeling" / "configs" / "0905_cfg.yaml")
    rasterizer = build_rasterizer(cfg, dm)

    faster_agent_dataset = FasterAgentDataset(cfg, dataset, rasterizer, None)
    fast_agent_dataset = FastAgentDataset(cfg, dataset, rasterizer, None)

    assert len(faster_agent_dataset) == len(fast_agent_dataset)
    keys = ["image", "target_positions", "target_availabilities"]
    for index in tqdm(range(min(1000, len(faster_agent_dataset)))):
        actual = faster_agent_dataset[index]
        expected = fast_agent_dataset[index]
        for key in keys:
            assert (actual[key] == expected[key]).all()
