import random
import numpy as np
import torch
from augmentations import mask_keypoints, zero_keypoints, shift_keypoints
from matching.utils import format_keypoints


class TrainInteDataset:
    def __init__(self, annotations, batch_size=64):
        # Image properties
        self.width = 640
        self.height = 360
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Ground truth annotation for loss calculation
        self.ground_truth = []

        # Visibility mask
        self.masks = []

        # Pair of poses
        self.p1 = []
        self.p2 = []

        # Number of batches
        batches = len(annotations) * 2 // batch_size

        augmentation_funcs = [mask_keypoints, zero_keypoints, shift_keypoints]

        for pose in annotations:
            keypoints = pose["keypoints"]
            ground_truth_keypoints = keypoints.copy()

            mask = []

            # Setting confidence to 1 for ground truth
            for i in range(len(ground_truth_keypoints) // 3):
                # Saving visibility mask
                mask.append(ground_truth_keypoints[2 + (i * 3)] // 2)
                ground_truth_keypoints[2 + (i * 3)] = 1
            self.masks.extend([mask, mask])

            self.ground_truth.append(format_keypoints(ground_truth_keypoints))
            self.ground_truth.append(format_keypoints(ground_truth_keypoints))

            p = random.random()

            if p < 0.1:
                aug1, aug2, aug3 = random.sample(augmentation_funcs, 3)

                # First pair: ground truth + random augmentation 1
                self.p1.append(format_keypoints(ground_truth_keypoints))
                self.p2.append(format_keypoints(aug1(ground_truth_keypoints.copy())))

                # Second pair: other 2 augmentations
                self.p1.append(format_keypoints(aug2(ground_truth_keypoints.copy())))
                self.p2.append(format_keypoints(aug3(ground_truth_keypoints.copy())))

            else:
                aug1, aug2, aug3 = random.sample(augmentation_funcs, 3)
                # First pair: two random augmentations
                self.p1.append(format_keypoints(aug1(ground_truth_keypoints.copy())))
                self.p2.append(format_keypoints(aug2(ground_truth_keypoints.copy())))

                # Second pair: random augmentation from both selected above with other unused one
                self.p1.append(format_keypoints(aug3(ground_truth_keypoints.copy())))
                self.p2.append(
                    format_keypoints(
                        random.choice([aug1, aug2])(ground_truth_keypoints.copy())
                    )
                )

        # Divide into batches
        self.ground_truth = np.array_split(np.array(self.ground_truth), batches)
        self.p1 = np.array_split(np.array(self.p1), batches)
        self.p2 = np.array_split(np.array(self.p2), batches)

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return len(self.ground_truth)

    def __next__(self):
        if self.pos >= len(self.ground_truth):
            raise StopIteration
        ground_truth = self.ground_truth[self.pos]
        p1 = self.p1[self.pos]
        p2 = self.p2[self.pos]
        mask = self.masks[self.pos]

        ground_truth = np.float32(ground_truth)
        p2 = np.float32(p2)

        source_pts = np.stack([p1, p2], axis=1)
        source_pts = source_pts.reshape([source_pts.shape[0], -1])

        self.pos += 1

        source_pts = torch.tensor(source_pts).to(self.device)
        ground_truth = torch.tensor(ground_truth).to(self.device)
        mask = torch.tensor(mask).to(self.device)

        return source_pts, ground_truth, mask
