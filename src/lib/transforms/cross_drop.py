from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations.augmentations.functional as F
import random

class CrossDrop(ImageOnlyTransform):

    def __init__(self, max_h_cut=0.2, max_w_cut=0.2, fill_value=0, always_apply=False, p=0.5):
        super(CrossDrop, self).__init__(always_apply, p)
        self.max_h_cut = max_h_cut
        self.max_w_cut = max_w_cut
        self.fill_value = fill_value

    def apply(self, image, fill_value=0, holes=(), **params):
        return F.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        y1 = int(random.random() * self.max_h_cut * height)
        x1 = int(random.random() * self.max_w_cut * width)

        y2 = int(random.random() * self.max_h_cut * height)
        x2 = int(random.random() * self.max_w_cut * width)

        y3 = int(random.random() * self.max_h_cut * height)
        x3 = int(random.random() * self.max_w_cut * width)
            
        y4 = int(random.random() * self.max_h_cut * height)
        x4 = int(random.random() * self.max_w_cut * width)

        return {"holes": [
            (0, 0, x1, y1),
            (width-x2, 0, width, y2),
            (0, height-y3, x3, height),
            (width-x4, height-y4, width, height)
        ]}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("max_h_cut", "max_w_cut")
