import json
import random
import numpy as np
from augmentations import mask_keypoints, zero_keypoints, shift_keypoints


class TrainInteDataset:

    def __init__(self, train_path, batch_size=64):
        f = open(train_path)
        data = json.load(f)
        f.close()

        # Ground truth annotation for loss calculation
        self.ground_truth = []

        # Pair of poses
        self.p1 = []
        self.p2 = []

        annotations = data["annotations"]

        # Number of batches
        batches = len(annotations) // batch_size

        augmentation_funcs = [mask_keypoints, zero_keypoints, shift_keypoints]

        for pose in annotations:
            keypoints = pose["keypoints"]
            ground_truth_keypoints = keypoints.copy()

            # Setting confidence to 1 for ground truth
            for i in range(len(ground_truth_keypoints) // 3):
                ground_truth_keypoints[2 + (i * 3)] = 1
            self.ground_truth.append(
                self.format_keypoints(ground_truth_keypoints))

            p = random.random()

            if p < 0.1:
                self.p1.append(self.format_keypoints(ground_truth_keypoints))
                # Randomly choose an augmentation
                augmented_keypoints = random.choice(augmentation_funcs)(
                    ground_truth_keypoints.copy())
                self.p2.append(self.format_keypoints(augmented_keypoints))
            else:
                aug1, aug2 = random.choices(augmentation_funcs, k=2)
                self.p1.append(
                    self.format_keypoints(aug1(ground_truth_keypoints.copy())))
                self.p2.append(
                    self.format_keypoints(aug2(ground_truth_keypoints.copy())))

        #Divide into batches
        self.ground_truth = np.array_split(
            np.array(self.ground_truth), batches)
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

        ground_truth = np.float32(ground_truth)
        p2 = np.float32(p2)

        source_pts = np.stack([p1, p2], axis=1)
        source_pts = source_pts.reshape([source_pts.shape[0], -1])

        self.pos += 1

        return source_pts, ground_truth

    # From matching/utils.py
    def format_keypoints(self, keypoints):
        """
        Format keypoints
        """
        x_coord = []
        y_coord = []
        confidence = []
        split = np.split(np.array(keypoints), len(keypoints) // 3)

        for x, y, c in split:
            x_coord.append(x)
            y_coord.append(y)
            confidence.append(c)

        return np.array(
            [np.array(x_coord),
             np.array(y_coord),
             np.array(confidence)])
