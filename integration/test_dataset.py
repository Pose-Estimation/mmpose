import json
import random
import numpy as np
import torch
from augmentations import mask_keypoints, zero_keypoints, shift_keypoints


class TestInteDataset:

    def __init__(self, ground_truth, bottom_up, top_down, batch_size=64):
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
        self.bottom_up_kpts = []
        self.top_down_kpts = []

        # Number of batches
        batches = len(annotations) * 2 // batch_size

        augmentation_funcs = [mask_keypoints, zero_keypoints, shift_keypoints]

        for pose in ground_truth:
            ground_truth_keypoints = pose["keypoints"]

            mask = []

            # Setting confidence to 1 for ground truth
            for i in range(len(ground_truth_keypoints) // 3):
                # Saving visibility mask
                mask.append(ground_truth_keypoints[2 + (i * 3)] // 2)
                ground_truth_keypoints[2 + (i * 3)] = 1
            self.masks.extend([mask, mask])

            self.ground_truth.append(
                self.format_keypoints(ground_truth_keypoints))

        for pose in bottom_up:
            bottom_up_keypoints = pose["keypoints"]
             self.bottom_up.append(
                self.format_keypoints(bottom_up_keypoints))

        for pose in top_down:
            top_down_keypoints = pose["keypoints"]
             self.top_down.append(
                self.format_keypoints(top_down_keypoints))

        # Divide into batches
        self.ground_truth = np.array_split(
            np.array(self.ground_truth), batches)
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

    # From matching/utils.py
    def format_keypoints(self, keypoints):
        """
        Format keypoints
        """
        x_coord = []
        y_coord = []
        # confidence = []
        split = np.split(np.array(keypoints), len(keypoints) // 3)

        # Creating individual x,y arrays and normalizing the values based on image sizes
        for x, y, _ in split:
            x_coord.append(x / self.width)
            y_coord.append(y / self.height)
            # confidence.append(c)

        return np.array([np.array(x_coord), np.array(y_coord)])
