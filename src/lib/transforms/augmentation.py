from albumentations import Compose
import albumentations as A
import numpy as np
from lib.transforms.cross_drop import CrossDrop

from lib.rasterization.agent_type_box_rasterizer import CAR_LABEL_INDEX, CYCLIST_LABEL_INDEX, PEDESTRIAN_LABEL_INDEX

label_id_to_index = {
    CAR_LABEL_INDEX: 0,
    CYCLIST_LABEL_INDEX: 1,
    PEDESTRIAN_LABEL_INDEX: 2,
}


def _agent_type_onehot(label_probabilities):
    label = np.argmax(label_probabilities)

    x_feat = np.array([0, 0, 0], dtype=np.float32)
    x_feat[label_id_to_index[label]] = 1.0
    return x_feat


class ImageAugmentation(object):

    def __init__(self, flags):
        self.aug = None
        self.set_augmentation(flags)

        self.feat_mode = flags.feat_mode

    def set_augmentation(self, flags):
        aug_list = []
        if flags.blur["p"] > 0.0:
            aug_list.append(A.Blur(**flags.blur))
        if flags.cutout["p"] > 0.0:
            aug_list.append(A.Cutout(**flags.cutout))
        if flags.downscale["p"] > 0.0:
            aug_list.append(A.Downscale(**flags.downscale))
        if flags.crossdrop["p"] > 0.0:
            aug_list.append(CrossDrop(**flags.crossdrop))
        self.aug = Compose(aug_list) if len(aug_list)!=0 else None

    def transform(self, batch):
        if self.aug is not None:
            image = batch["image"]
            image = self.aug(image=np.moveaxis(image, 0, 2))["image"]
            image = np.moveaxis(image, 2, 0)
            batch["image"] = image

        if self.feat_mode == "none":
            return batch["image"], batch["target_positions"], batch["target_availabilities"]
        elif self.feat_mode == "agent_type":
            x_feat = _agent_type_onehot(batch["label_probabilities"])
            return batch["image"], batch["target_positions"], batch["target_availabilities"], x_feat
