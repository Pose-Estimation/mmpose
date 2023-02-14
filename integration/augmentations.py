import math
from typing import List
import random
import numpy as np

PATH_TO_VIDEOPOSE = input(
    "Enter the absolute path to your video_pose/full_data directory:"
)
VIDEO_POSE_TYPES = {"No_penalty": 0, "Slashing": 1, "Tripping": 2}

# The number of joints to occlude/shift. As we experiment and try to improve our model, we may need to adjust
# these values experimentally.
MIN_JOINTS_TO_MASK = 3
MAX_JOINTS_TO_MASK = 6
MIN_JOINTS_TO_SHIFT = 4
MAX_JOINTS_TO_SHIFT = 8

# For the third augmentation, this is the odds that the first player in an image will be zeroed. If this value is set to
# 0.5, then there is a 50% chance the player will be zeroed. Initially, this value is set to 1 so that zeroing will always
# happen. Need to experiment to determine if this is best, since
ODDS_TO_ZERO = 1


############### MASKING ################
def mask_keypoints(player_keypoints) -> List[List[int]]:
    """
    Masking
    """
    # the number of joints to mask, randomly between 3 and 5
    num_to_mask = random.randint(MIN_JOINTS_TO_MASK, MAX_JOINTS_TO_MASK)
    # which joints are masked
    joints_to_mask = random.sample(range(14), num_to_mask)
    # for each of those joints, set all three variables to 0
    confidence = random.randint(0, 200) / 1000  # Random confidence from 0.000 to 0.200
    for joint_index in joints_to_mask:
        # Confidence
        player_keypoints[joint_index * 3 + 2] = confidence
        # Mask
        player_keypoints[joint_index * 3] = 0
        player_keypoints[joint_index * 3 + 1] = 0
        player_keypoints[joint_index * 3 + 2] = 0
    return player_keypoints


############### SHIFTING ################

# Maximum value observed over multiple runs of this function
BIGGEST_SHIFT = 14000


def shift_keypoints(player_keypoints) -> List[List[int]]:
    """
    Shifting
    """
    # the number of joints to shift, randomly between 3 and 5
    num_to_shift = random.randint(MIN_JOINTS_TO_SHIFT, MAX_JOINTS_TO_SHIFT)
    # which joints are shifted
    joints_to_shift = random.sample(range(14), num_to_shift)

    # TODO what to do if nose or neck is occluded? what distance to use?
    nosex = player_keypoints[0]
    nosey = player_keypoints[1]
    # nose_certainty = player_keypoints[2]
    neckx = player_keypoints[3]
    necky = player_keypoints[4]
    # neck_certainty = player_keypoints[5]

    # distance from nose to neck
    half_face_distance = math.sqrt(
        (nosex - neckx) * (nosex - neckx) + (nosey - necky) * (nosey - necky)
    )
    # random.random() is in range [0,1), adding 1 puts it in range [1,2), multiplying by halfFaceDistance puts it in range [hFD, 2*hFD)
    dist_to_shift = half_face_distance * (random.random() + 1)

    # Assigning confidence values based on size of shift
    # The bigger the shift the smaller the confidence value
    # Between 0.3 and 0.7
    percentage = 1.0 - dist_to_shift / BIGGEST_SHIFT
    confidence = 0.3 + 0.4 * percentage

    # 0 = +x; 1 = -x; 2 = +y; 3 = -y
    dir_to_shift = random.randint(0, 3)

    # TODO currently the same shift (distance and direction) is used for each of the 4-8 joints. Is it better to randomly use different
    # shifts? Will need to experiment. Can do so simply by moving the RNG for distToShift and dirToShift into the for loop below
    for joint_index in joints_to_shift:
        # Confidence
        player_keypoints[joint_index * 3 + 2] = confidence
        # Shift
        if dir_to_shift == 0:
            player_keypoints[joint_index * 3] += dist_to_shift
        elif dir_to_shift == 0:
            player_keypoints[joint_index * 3] -= dist_to_shift
        elif dir_to_shift == 0:
            player_keypoints[joint_index * 3 + 1] += dist_to_shift
        elif dir_to_shift == 3:
            player_keypoints[joint_index * 3 + 1] -= dist_to_shift
    return player_keypoints


############### ZEROING ################
def zero_keypoints(player_keypoints) -> List[List[int]]:
    """
    Zeroing
    """
    # TODO Currently, the first player in each image is being zeroed. This was done out of simplicity, to not
    # need to figure out how many players are in an image. Will it improve performance if the player that's
    # zeroed is randomly chosen? If so, will need to figure out how.

    # Check if we've gotten to a new image; if so, potentially zero the first player depending on the odds
    if random.random() < ODDS_TO_ZERO:
        np.zeros(len(player_keypoints))

    confidence = random.randint(0, 200) / 1000  # Random confidence from 0.000 to 0.200
    for joint_index in range(len(player_keypoints) // 3):
        # Confidence
        player_keypoints[joint_index * 3 + 2] = confidence

    return player_keypoints
