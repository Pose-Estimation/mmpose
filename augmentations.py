import math
import numpy as np
import json
import sys
import os
import random

PATH_TO_VIDEOPOSE = input("Enter the absolute path to your video_pose directory:")
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

#get image width and height, assuming they are the same size for all images in one video
#Note: even now that it's images from multiple videos, it seems like all images in the dataset
#are 640x360
imageWidth = data["images"][0]["width"]
imageHeight = data["images"][0]["height"]

masking_data = data.copy()
shifting_data = data.copy()
zeroing_data = data.copy()

############### MASKING ################
for player in masking_data["annotations"] :
    playerKeypoints = player["keypoints"]

    # the number of joints to mask, randomly between 3 and 5
    numToMask = random.randint(MIN_JOINTS_TO_MASK,MAX_JOINTS_TO_MASK)
    # which joints are masked
    jointsToMask = random.sample(range(14), numToMask)
    # for each of those joints, set all three variables to 0
    for i in jointsToMask :
        playerKeypoints[i * 3] = 0
        playerKeypoints[i * 3 + 1] = 0
        playerKeypoints[i * 3 + 2] = 0
maskingPath = f"{PATH_TO_VIDEOPOSE}/train/masking-coco.json"
maskingFile = open(maskingPath, "w")
maskingJson = json.dump(masking_data, maskingFile, indent=4)
maskingFile.close()

############### SHIFTING ################
for player in shifting_data["annotations"] :
    playerKeypoints = player["keypoints"]
    # the number of joints to shift, randomly between 3 and 5
    numToShift = random.randint(MIN_JOINTS_TO_SHIFT,MAX_JOINTS_TO_SHIFT)
    # which joints are shifted
    jointsToShift = random.sample(range(14), numToShift)

    # TODO what to do if nose or neck is occluded? what distance to use?
    nosex = playerKeypoints[0]
    nosey = playerKeypoints[1]
    noseCertainty = playerKeypoints[2]
    neckx = playerKeypoints[3]
    necky = playerKeypoints[4]
    neckCertainty = playerKeypoints[5]

    #distance from nose to neck
    halfFaceDistance = math.sqrt((nosex - neckx) * (nosex - neckx) + (nosey - necky) * (nosey - necky))
    # random.random() is in range [0,1), adding 1 puts it in range [1,2), multiplying by halfFaceDistance puts it in range [hFD, 2*hFD)
    distToShift = halfFaceDistance * (random.random() + 1)
    # 0 = +x; 1 = -x; 2 = +y; 3 = -y
    dirToShift = random.randint(0,3)

    # TODO currently the same shift (distance and direction) is used for each of the 4-8 joints. Is it better to randomly use different
    # shifts? Will need to experiment. Can do so simply by moving the RNG for distToShift and dirToShift into the for loop below
    for joint in jointsToShift:
        if (dirToShift == 0) :
            playerKeypoints[joint * 3] += distToShift
        elif (dirToShift == 0) :
            playerKeypoints[joint * 3] -= distToShift
        elif (dirToShift == 0) :
            playerKeypoints[joint * 3 + 1] += distToShift
        elif (dirToShift == 3) :
            playerKeypoints[joint * 3 + 1] -= distToShift
shiftingPath = f"{PATH_TO_VIDEOPOSE}/train/shifting-coco.json"
shiftingFile = open(shiftingPath, "w")
shiftingJson = json.dump(shifting_data, shiftingFile, indent=4)
shiftingFile.close()

############### ZEROING ################
refImageId = -1
for player in zeroing_data["annotations"] :
    # TODO Currently, the first player in each image is being zeroed. This was done out of simplicity, to not
    # need to figure out how many players are in an image. Will it improve performance if the player that's 
    # zeroed is randomly chosen? If so, will need to figure out how.

    # Check if we've gotten to a new image; if so, potentially zero the first player depending on the odds
    if (refImageId != player["image_id"]) :
        refImageId = player["image_id"]
        if (random.random() < ODDS_TO_ZERO) :
            playerKeypoints = [0] * 42
zeroingPath = f"{PATH_TO_VIDEOPOSE}/train/zeroing-coco.json"
zeroingFile = open(zeroingPath, "w")
zeroingJson = json.dump(zeroing_data, zeroingFile, indent=4)
zeroingFile.close()