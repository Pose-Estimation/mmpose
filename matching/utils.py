import numpy as np


def format_keypoints(keypoints):
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