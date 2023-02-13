import json
import numpy as np
from augmentations import mask_keypoints, zero_keypoints, shift_keypoints


class TrainInteDataset():

    def __init__(self, train_path):
        f = open(train_path)
        data = json.load(f)
        f.close()
        
        self.ground_truth = []
        self.augmentations = []
        self.aug_func = [mask_keypoints, zero_keypoints, shift_keypoints]

        #TODO create dataset
        annotations = data["annotations"]

        for pose in annotations:
            
            # TODO Set all visibility to 1
            # Apply random augmentation
            
            pass

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
