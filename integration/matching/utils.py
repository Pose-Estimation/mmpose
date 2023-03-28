import numpy as np 


def format_keypoints(keypoints, width, height, confidence=False):
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

    if confidence:
        return np.array(
            [np.array(x_coord),
             np.array(y_coord),
             np.array(confidence)])

    return np.array([np.array(x_coord), np.array(y_coord)])


def format_keypoints_mask(keypoints, width, height):
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
        mask.append(v // 2)

    return np.array([np.array(x_coord), np.array(y_coord)]), np.array(mask)
