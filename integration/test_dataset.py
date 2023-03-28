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
        self.bottom_up_kpts = [format_keypoints(
            kpt, self.width, self.width) for kpt in bottom_up_kpts]
        self.top_down_kpts = []

        self.img_id = []

        # Number of batches
        # batches = len(bottom_up_kpts) // batch_size

        for pose in top_down_annots:
            top_down_keypoints = pose["keypoints"]
            self.img_id.append(pose["image_id"])
            self.top_down.append(format_keypoints(
                top_down_keypoints, self.width, self.height))

        # Divide into batches
        # self.bottom_up_kpts = np.array_split(
        #     np.array(self.bottom_up_kpts), batches)
        # self.top_down_kpts = np.array_split(
        #     np.array(self.top_down_kpts), batches)

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return len(self.bottom_up_kpts)

    def __next__(self):
        if self.pos >= len(self.bottom_up_kpts):
            raise StopIteration
        bottom_up_kpts = self.bottom_up_kpts[self.pos]
        top_down_kpts = self.top_down_kpts[self.pos]
        img_id = self.img_id[self.pos]

        top_down_kpts = np.float32(top_down_kpts)

        source_pts = np.stack([bottom_up_kpts, top_down_kpts], axis=1)
        source_pts = source_pts.reshape([source_pts.shape[0], -1])

        self.pos += 1

        source_pts = torch.tensor(source_pts).to(self.device)

        return source_pts, img_id
