import json
import random
import numpy as np
from augmentations import mask_keypoints, zero_keypoints, shift_keypoints


class TrainInteDataset:
    def __init__(self, train_path):
        f = open(train_path)
        data = json.load(f)
        f.close()

        self.ground_truth = []
        self.augmentations = []
        self.aug_func = [mask_keypoints, zero_keypoints, shift_keypoints]

        # TODO create dataset
        annotations = data["annotations"]
        augmentation_funcs = [mask_keypoints, zero_keypoints, shift_keypoints]
        for pose in annotations:
            keypoints = pose[keypoints]
            ground_truth_keypoints = keypoints.copy()

            # Setting confidence to 1 for ground truth
            for i in range(len(pose) // 3):
                ground_truth_keypoints[2 + (i * 3)] = 1
            self.ground_truth.append(ground_truth_keypoints)

            # Randomly choose an augmentation
            augmented_keypoints = random.choice(augmentation_funcs)(
                ground_truth_keypoints.copy()
            )
            self.augmentations.append(augmented_keypoints)

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return len(self.ground_truth)

    def __next__(self):
        if self.pos >= len(self.ground_truth):
            raise StopIteration
        ground_truth = self.ground_truth[self.pos]
        augment = self.augmentations[self.pos]

        ground_truth = np.float32(ground_truth)
        augment = np.float32(augment)

        source_pts = np.stack([augment, ground_truth], axis=1)

        self.pos += 1

        return source_pts, ground_truth
