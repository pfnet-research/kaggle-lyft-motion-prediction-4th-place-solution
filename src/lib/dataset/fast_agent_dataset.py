import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer

from lib.utils.timer_utils import timer

# WARNING: changing these values impact the number of instances selected for both train and inference!

MIN_FRAME_HISTORY = 10  # minimum number of frames an agents must have in the past to be picked
MIN_FRAME_FUTURE = 1  # minimum number of frames an agents must have in the future to be picked


class FastAgentDataset(AgentDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
        agents_mask: Optional[np.ndarray] = None,
        min_frame_history: int = MIN_FRAME_HISTORY,
        min_frame_future: int = MIN_FRAME_FUTURE,
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"

        super(AgentDataset, self).__init__(cfg, zarr_dataset, rasterizer, perturbation)

        # store the valid agents indexes
        with timer("agents_indices"):
            self.agents_indices = self.load_agents_indices(agents_mask, min_frame_history, min_frame_future)
        print("self.agents_indices", self.agents_indices.shape)
        # this will be used to get the frame idx from the agent idx
        with timer("cumulative_sizes_agents"):
            self.cumulative_sizes_agents = self.load_cumulative_sizes_agents(agents_mask)
        print("self.cumulative_sizes_agents", self.cumulative_sizes_agents.shape)

        # agents_mask may be `None` here.
        # Because this is not used in typical training & takes time to load...
        self.agents_mask = agents_mask

    # --- Below 2 methods are for "Fast" __init__. Caching time consuming array loading.
    def load_agents_indices(
            self,
            agents_mask: Optional[np.ndarray] = None,
            min_frame_history: int = MIN_FRAME_HISTORY,
            min_frame_future: int = MIN_FRAME_FUTURE,
    ) -> np.ndarray:
        agent_prob = self.cfg["raster_params"]["filter_agents_threshold"]
        agents_mask_str = "" if agents_mask is None else f"_mask{np.sum(agents_mask)}"
        filename = f"agents_indices_{agent_prob}_{min_frame_history}_{min_frame_future}{agents_mask_str}.npz"
        agents_indices_path = Path(self.dataset.path) / filename
        if not agents_indices_path.exists():
            print(f"Cache {agents_indices_path} does not exist, creating...")
            if agents_mask is None:  # if not provided try to load it from the zarr
                with timer("load_agents_mask"):
                    agents_mask = self.load_agents_mask()
                print("agents_mask", agents_mask.shape)
                past_mask = agents_mask[:, 0] >= min_frame_history
                future_mask = agents_mask[:, 1] >= min_frame_future
                agents_mask = past_mask * future_mask

                if min_frame_history != MIN_FRAME_HISTORY:
                    warnings.warn(
                        f"you're running with custom min_frame_history of {min_frame_history}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if min_frame_future != MIN_FRAME_FUTURE:
                    warnings.warn(
                        f"you're running with custom min_frame_future of {min_frame_future}", RuntimeWarning, stacklevel=2
                    )
            else:
                warnings.warn("you're running with a custom agents_mask", RuntimeWarning, stacklevel=2)
            agents_indices = np.nonzero(agents_mask)[0]
            np.savez_compressed(str(agents_indices_path), agents_indices=agents_indices)

        # --- Load from cache ---
        assert agents_indices_path.exists()
        agents_indices = np.load(agents_indices_path)["agents_indices"]
        print(f"Loaded from {agents_indices_path}")
        return agents_indices

    def load_cumulative_sizes_agents(self, agents_mask: Optional[np.ndarray] = None) -> np.ndarray:
        agents_mask_str = "" if agents_mask is None else f"_mask{np.sum(agents_mask)}"
        filename = f"cumulative_sizes_agents{agents_mask_str}.npz"
        cumulative_sizes_agents_path = Path(self.dataset.path) / filename
        if not cumulative_sizes_agents_path.exists():
            print(f"Cache {cumulative_sizes_agents_path} does not exist, creating...")
            cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]
            np.savez_compressed(str(cumulative_sizes_agents_path), cumulative_sizes_agents=cumulative_sizes_agents)

        # --- Load from cache ---
        assert cumulative_sizes_agents_path.exists()
        cumulative_sizes_agents = np.load(cumulative_sizes_agents_path)["cumulative_sizes_agents"]
        print(f"Loaded from {cumulative_sizes_agents_path}")
        return cumulative_sizes_agents

    def get_scene_dataset(self, scene_index: int) -> "AgentDataset":
        """
        Differs from parent only in the return type.
        Instead of doing everything from scratch, we rely on super call and fix the agents_mask
        """
        if self.agents_mask is None:
            print("Loading agents_mask...")
            self.agents_mask = self.load_agents_mask()
        return super(FastAgentDataset, self).get_scene_dataset(scene_index)
