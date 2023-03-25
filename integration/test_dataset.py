import numpy as np
import torch
from matching.utils import format_keypoints


class TestInteDataset:
    def __init__(
        self, ground_truth_annots, bottom_up_kpts, top_down_annots, batch_size=64
    ):
        # Image properties
        self.width = 640
        self.height = 360
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Ground truth annotation for loss calculation
        self.ground_truth = []

        # Visibility mask
        self.masks = []

        # Pair of poses
        self.bottom_up_kpts = bottom_up_kpts
        self.top_down_kpts = []

        # Number of batches
        batches = len(ground_truth_annots) * 2 // batch_size

        for pose in ground_truth_annots:
            ground_truth_keypoints = pose["keypoints"]

            mask = []

            # Setting confidence to 1 for ground truth
            for i in range(len(ground_truth_keypoints) // 3):
                # Saving visibility mask
                mask.append(ground_truth_keypoints[2 + (i * 3)] // 2)
                ground_truth_keypoints[2 + (i * 3)] = 1
            self.masks.extend([mask, mask])

            self.ground_truth.append(format_keypoints(ground_truth_keypoints))

        for pose in top_down_annots:
            top_down_keypoints = pose["keypoints"]
            self.top_down.append(format_keypoints(top_down_keypoints))

        # Divide into batches
        self.ground_truth = np.array_split(np.array(self.ground_truth), batches)
        self.bottom_up_kpts = np.array_split(np.array(self.bottom_up_kpts), batches)
        self.top_down_kpts = np.array_split(np.array(self.top_down_kpts), batches)

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return len(self.ground_truth)

    def __next__(self):
        if self.pos >= len(self.ground_truth):
            raise StopIteration
        ground_truth = self.ground_truth[self.pos]
        bottom_up_kpts = self.bottom_up_kpts[self.pos]
        top_down_kpts = self.top_down_kpts[self.pos]
        mask = self.masks[self.pos]

        ground_truth = np.float32(ground_truth)
        top_down_kpts = np.float32(top_down_kpts)

        source_pts = np.stack([bottom_up_kpts, top_down_kpts], axis=1)
        source_pts = source_pts.reshape([source_pts.shape[0], -1])

        self.pos += 1

        source_pts = torch.tensor(source_pts).to(self.device)
        ground_truth = torch.tensor(ground_truth).to(self.device)
        mask = torch.tensor(mask).to(self.device)

        return source_pts, ground_truth, mask
