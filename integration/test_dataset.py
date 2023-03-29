import numpy as np
import torch
from matching.utils import format_keypoints


class TestInteDataset:

    def __init__(self, bottom_up_kpts, top_down_annots, batch_size=64):
        # Image properties
        self.width = 640
        self.height = 360
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # Ground truth annotation for loss calculation
        self.ground_truth = []

        # Visibility mask
        self.masks = []

        # Pair of poses
        self.bottom_up_kpts = bottom_up_kpts
        self.top_down_kpts = []

        self.img_id = []

        for pose in top_down_annots:
            top_down_keypoints = pose["keypoints"]
            self.img_id.append(pose["image_id"])
            self.top_down_kpts.append(
                format_keypoints(top_down_keypoints, self.width, self.height))

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return len(self.bottom_up_kpts)

    def __next__(self):
        if self.pos >= len(self.bottom_up_kpts):
            raise StopIteration
        bottom_up_kpts = np.float32(self.bottom_up_kpts[self.pos])
        top_down_kpts = np.float32(self.top_down_kpts[self.pos])
        img_id = self.img_id[self.pos]

        source_pts = np.stack([bottom_up_kpts, top_down_kpts], axis=1)
        source_pts = source_pts.reshape([-1, 56])

        self.pos += 1

        source_pts = torch.tensor(source_pts).to(self.device)

        return source_pts, img_id
