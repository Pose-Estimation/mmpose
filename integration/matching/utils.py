from collections import defaultdict
import json
import numpy as np


def format_keypoints(keypoints, width=640, height=360, use_confidence=False):
    """
    Format keypoints
    """
    x_coord = []
    y_coord = []
    confidence = []
    split = np.split(np.array(keypoints), len(keypoints) // 3)

    for x, y, c in split:
        x_coord.append(x / width)
        y_coord.append(y / height)
        confidence.append(c)

    if use_confidence:
        return np.array(
            [np.array(x_coord),
             np.array(y_coord),
             np.array(confidence)])

    return np.array([np.array(x_coord), np.array(y_coord)])


def format_keypoints_mask(keypoints):
    """
    Format keypoints with mask
    """
    x_coord = []
    y_coord = []
    mask = []
    split = np.split(np.array(keypoints), len(keypoints) // 3)

    for x, y, v in split:
        x_coord.append(x)
        y_coord.append(y)
        mask.append(True if v // 2 else False)

    return np.array([np.array(x_coord), np.array(y_coord)]), np.array(mask)


def format_annotations(path, confidence=False):
    # Load predictions
    estimations = defaultdict(list)
    predictions = json.load(open(path))
    for pred in predictions:
        image_id = pred["image_id"]
        estimations[image_id].append(
            format_keypoints(pred["keypoints"], 1, 1, confidence))

    return estimations
