import math
import numpy as np
import json
import sys
import os
import random
from tqdm import tqdm
import pickle
import glob

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

training_file_path = f"{PATH_TO_VIDEOPOSE}/train/train-coco.json"
f = open(training_file_path)
data = json.load(f)
f.close()

# get image width and height, assuming they are the same size for all images in one video
# Note: even now that it's images from multiple videos, it seems like all images in the dataset
# are 640x360
imageWidth = data["images"][0]["width"]
imageHeight = data["images"][0]["height"]

masking_data = data.copy()
shifting_data = data.copy()
zeroing_data = data.copy()


# function to format keypoints in required format for integration network
def format_keypoints(keypoints):
    x_coord = []
    y_coord = []
    confidence = []
    split = np.split(np.array(keypoints), len(keypoints) // 3)

    for x, y, c in split:
        x_coord.append(x)
        y_coord.append(y)
        confidence.append(c)

    return np.array([np.array(x_coord), np.array(y_coord), np.array(confidence)])


############### MASKING ################
for player in masking_data["annotations"]:
    playerKeypoints = player["keypoints"]

    # the number of joints to mask, randomly between 3 and 5
    numToMask = random.randint(MIN_JOINTS_TO_MASK, MAX_JOINTS_TO_MASK)
    # which joints are masked
    jointsToMask = random.sample(range(14), numToMask)
    # for each of those joints, set all three variables to 0
    for i in jointsToMask:
        playerKeypoints[i * 3] = 0
        playerKeypoints[i * 3 + 1] = 0
        playerKeypoints[i * 3 + 2] = 0
maskingPath = f"{PATH_TO_VIDEOPOSE}/train/masking-coco.json"
maskingFile = open(maskingPath, "w")
maskingJson = json.dump(masking_data, maskingFile, indent=4)
maskingFile.close()

############### SHIFTING ################
for player in shifting_data["annotations"]:
    playerKeypoints = player["keypoints"]
    # the number of joints to shift, randomly between 3 and 5
    numToShift = random.randint(MIN_JOINTS_TO_SHIFT, MAX_JOINTS_TO_SHIFT)
    # which joints are shifted
    jointsToShift = random.sample(range(14), numToShift)

    # TODO what to do if nose or neck is occluded? what distance to use?
    nosex = playerKeypoints[0]
    nosey = playerKeypoints[1]
    noseCertainty = playerKeypoints[2]
    neckx = playerKeypoints[3]
    necky = playerKeypoints[4]
    neckCertainty = playerKeypoints[5]

    # distance from nose to neck
    halfFaceDistance = math.sqrt(
        (nosex - neckx) * (nosex - neckx) + (nosey - necky) * (nosey - necky)
    )
    # random.random() is in range [0,1), adding 1 puts it in range [1,2), multiplying by halfFaceDistance puts it in range [hFD, 2*hFD)
    distToShift = halfFaceDistance * (random.random() + 1)
    # 0 = +x; 1 = -x; 2 = +y; 3 = -y
    dirToShift = random.randint(0, 3)

    # TODO currently the same shift (distance and direction) is used for each of the 4-8 joints. Is it better to randomly use different
    # shifts? Will need to experiment. Can do so simply by moving the RNG for distToShift and dirToShift into the for loop below
    for joint in jointsToShift:
        if dirToShift == 0:
            playerKeypoints[joint * 3] += distToShift
        elif dirToShift == 0:
            playerKeypoints[joint * 3] -= distToShift
        elif dirToShift == 0:
            playerKeypoints[joint * 3 + 1] += distToShift
        elif dirToShift == 3:
            playerKeypoints[joint * 3 + 1] -= distToShift
shiftingPath = f"{PATH_TO_VIDEOPOSE}/train/shifting-coco.json"
shiftingFile = open(shiftingPath, "w")
shiftingJson = json.dump(shifting_data, shiftingFile, indent=4)
shiftingFile.close()

############### ZEROING ################
refImageId = -1
for player in zeroing_data["annotations"]:
    # TODO Currently, the first player in each image is being zeroed. This was done out of simplicity, to not
    # need to figure out how many players are in an image. Will it improve performance if the player that's
    # zeroed is randomly chosen? If so, will need to figure out how.

    # Check if we've gotten to a new image; if so, potentially zero the first player depending on the odds
    if refImageId != player["image_id"]:
        refImageId = player["image_id"]
        if random.random() < ODDS_TO_ZERO:
            playerKeypoints = [0] * 42
zeroingPath = f"{PATH_TO_VIDEOPOSE}/train/zeroing-coco.json"
zeroingFile = open(zeroingPath, "w")
zeroingJson = json.dump(zeroing_data, zeroingFile, indent=4)
zeroingFile.close()

print("Done augmenting...")

# choose two augmentation types randomly
aug_types = ["mask", "shift", "zero"]
chosen_augs = random.sample(aug_types, 2)

print("The random augmentations chosen are: " + str(chosen_augs))


# write function to concatenate two json files
def concatenateJson(file1, file2, outputFile):
    result = []

    with open(file1, "r") as infile:
        result.append(json.load(infile))
    with open(file2, "r") as infile:
        result.append(json.load(infile))

    all_images = []
    all_annotations = []
    for json_file in result:
        all_annotations += json_file["annotations"]
    all_images = result[1]["images"]
    jsonMerged = open(outputFile, "w")
    json.dump(
        {"images": all_images, "annotations": all_annotations}, jsonMerged, indent=4
    )
    jsonMerged.close()


augmented_file = f"{PATH_TO_VIDEOPOSE}/train/double_augmented-coco.json"  # <== this will be the file that is saved from concatenating
# concatenate results based on the two random augmentations chosen
if "mask" in chosen_augs and "shift" in chosen_augs:
    concatenateJson(maskingPath, shiftingPath, augmented_file)

elif "mask" in chosen_augs and "zero" in chosen_augs:
    concatenateJson(maskingPath, zeroingPath, augmented_file)

elif "shift" in chosen_augs and "zero" in chosen_augs:
    concatenateJson(shiftingPath, zeroingPath, augmented_file)

# modify json file into required format for integration network
f = open(augmented_file)
augmented_data = json.load(f)
f.close()

print("Done concatenating...")

# # # define directory to store the new file
# PATH_TO_AUGMENTED = input("Enter the absolute path in which you would like to store the augmented data:")

# # format the keypoints into pkl
# results = []
# prev = None
# for t in tqdm(augmented_data):
#     img_id = t['image_id']

#     if not (prev and prev == img_id):
#         formatted_kp = format_keypoints(t["keypoints"])
#         pickle.dump(results, open(os.path.join(PATH_TO_AUGMENTED, '%d.pkl' % img_id), 'wb'))
#         results.clear()
#         results.append(formatted_kp)

#     prev = img_id
