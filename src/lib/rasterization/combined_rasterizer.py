import numpy as np
from l5kit.rasterization import Rasterizer
from typing import List, Optional


class CombinedRasterizer(Rasterizer):
    def __init__(self, rasterzer_list: List[Rasterizer]):
        super(CombinedRasterizer, self).__init__()
        self.rasterzer_list = rasterzer_list
        try:
            self.raster_channels = sum([r.raster_channels for r in rasterzer_list])
        except:
            self.raster_channels = -1

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        image_list = [
            rasterizer.rasterize(history_frames, history_agents, history_tl_faces, agent)
            for rasterizer in self.rasterzer_list]
        return np.concatenate(image_list, -1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        try:
            ch_list = [rasterizer.raster_channels for rasterizer in self.rasterzer_list]
        except Exception as e:
            print("Need to set rasterizer.raster_channels attribute to use to_rgb!")
            raise e

        ch_split_indices = np.cumsum(ch_list)[:-1]
        image = None
        for i, im in enumerate(np.split(in_im, ch_split_indices, axis=-1)):
            this_rgb_image = self.rasterzer_list[i].to_rgb(im, **kwargs)
            if image is None:
                image = this_rgb_image
            else:
                # Overwrite this_rgb_image on top of image
                mask_box = np.any(this_rgb_image > 0, -1)
                image[mask_box] = this_rgb_image[mask_box]
        return image

    def __repr__(self):
        return "Combined: " + " + ".join([str(rasterizer) for rasterizer in self.rasterzer_list])
