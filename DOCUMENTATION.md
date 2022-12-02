# Pose Estimation

## Background
See official [mmpose documentation](https://mmpose.readthedocs.io/en/v0.29.0/) and [mmpose repo ReadMe](https://github.com/Pose-Estimation/mmpose/blob/master/README.md)

# Getting Started

## Script Order
1. convert_to_coco.py
1. convert_to_17.py
1. generate_dataset.py
1. generate_bounding_boxes.py

## 0. Config

**Associated script: hockey.py/hrnet_w32_hockey_512x512.py**
This training config uses many of the parameters from the original file while adapting it for a 14 keypoint dataset.

## 1. Convert from 18 to 17 keypoints

**Associated script: convert_to_17.py**

To convert the custom mmpose dataset keypoints to the correct COCO format keypoints, run the convert_to_17.py script. The script will prompt you to enter the absolute path to the video_pose directory on your local machine.

This script ignores 2 keypoints found in the inital dataset: the hockey grip and the hockey hill. It reorders the joints to follow the same format as the original coco dataset. The script also calculates new coordinates for the ears and eyes based on the neck and head position.

Afterwards, the script will create jsons for each video in each video folder in the video_pose directory, with the same name as the existing jsons with -json appended to it.

## 2. Convert to COCO

**Associated script: convert_to_coco.py**

To convert the custom mmpose dataset to the COCO format, run the convert_to_coco.py script. The script will prompt you to enter the absolute path to the video_pose directory on your local machine.
Afterwards, the script will create jsons for each video in each video folder in the video_pose directory, with the same name as the existing jsons with -json appended to it.

## 3. Generate Dataset

**Associated script: generate_dataset.py**
The script will prompt you to enter the absolute path to the video_pose directory on your local machine. The script then takes the file structure of the original dataset Dataset/penaltytype/gamedate and moves images to a new file structure based on a 70/15/15 split for the training/testing/validation sets. A new json file in the coco format is also created, however no bounding boxes are included therefore it is necessary to run the bounding box script on this directory.

## 4. Generate Bounding Boxes

**Associated script: generate_bounding_boxes.py**
To convert generate the bounding boxes, run the generate_bounding-boxes.py script. The script will prompt you to enter the absolute path to the video_pose directory on your local machine. Afterwards, the script will create two jsons for each video in each video folder in the video_pose directory, with the same name as the existing jsons with -bbox-appended.json or -bbox-only.json appended to it.

## 5. hrnet_w48_coco_256x192.py

**Associated script: configs\body\2d_kpt_sview_rgb_img\topdown_heatmap\coco\hrnet_w48_coco_256x192.py**
This training config uses many of the parameters from the original file while adapting it for a 14 keypoint dataset.

# Results

## Bottom up inference

### Initial Model
|  Game               |  AP       | AP .5      | AP .75     |  AR        | AR .5      |
|---------------------|-----------|------------|------------|------------|------------|
| Test Set            | 0.570     | 0.875      | 0.675      |  0.619     | 0.895      |
| 10/11/2017 CAR-CBJ  | 0.569     | 0.942      | 0.569      |  0.616     | 0.969      |
| 14/11/2017 BUF-PIT  | 0.495     | 0.882      | 0.477      |  0.552     | 0.912      |
| 24/11/2017 LAK-ARI  | 0.462     | 0.760      | 0.528      |  0.523     | 0.792      |
| 22/11/2017 OTT-WSH  | 0.592     | 0.941      | 0.649      |  0.647     | 0.957      |
| 18/11/2017 NYI-TBL  | 0.614     | 0.938      | 0.758      |  0.656     | 0.857      |
| 22/11/2017 MTL-NSH  | 0.648     | 0.983      | 0.834      |  0.706     | 0.996      |
| 25/11/2017 NYI-OTT  | 0.479     | 0.939      | 0.394      |  0.539     | 0.955      |
| 5/11/2017 NJD-CGY   | 0.357     | 0.662      | 0.417      |  0.405     | 0.714      |
| 7/11/2017 NJD-STL   | 0.391     | 0.853      | 0.283      |  0.438     | 0.862      |
| 8/11/2017 MIN-TOR   | 0.722     | 0.990      | 0.862      |  0.752     | 0.995      |
| 19/11/2017 COL-DET  | 0.339     | 0.821      | 0.330      |  0.407     | 0.854      |
| 18/11/2017 PIT-COL  | 0.434     | 0.693      | 0.546      |  0.462     | 0.709      |
### Fine Tuned Model
|  Game               |  AP       | AP .5      | AP .75     |  AR        | AR .5      |
|---------------------|-----------|------------|------------|------------|------------|
| Test Set            | 0.684     | 0.969      | 0.781      |  0.770     | 0.983      |
| 10/11/2017 CAR-CBJ  | 0.782     | 0.980      | 0.905      |  0.817     | 0.985      |
| 14/11/2017 BUF-PIT  | 0.773     | 0.999      | 0.957      |  0.806     | 1.00       |
| 24/11/2017 LAK-ARI  | 0.527     | 0.897      | 0.589      |  0.619     | 0.949      |
| 22/11/2017 OTT-WSH  | 0.736     | 0.993      | 0.842      |  0.797     | 1.00       |
| 18/11/2017 NYI-TBL  | 0.794     | 0.968      | 0.857      |  0.824     | 0.973      |
| 22/11/2017 MTL-NSH  | 0.781     | 0.996      | 0.934      |  0.831     | 1.00       |
| 25/11/2017 NYI-OTT  | 0.730     | 1.00       | 0.896      |  0.786     | 1.00       |
| 5/11/2017 NJD-CGY   | 0.703     | 0.955      | 0.829      |  0.755     | 0.967      |
| 7/11/2017 NJD-STL   | 0.733     | 0.964      | 0.921      |  0.777     | 0.977      |
| 8/11/2017 MIN-TOR   | 0.833     | 1.00       | 0.932      |  0.866     | 1.00       |
| 19/11/2017 COL-DET  | 0.710     | 1.00       | 0.874      |  0.766     | 1.00       |
| 18/11/2017 PIT-COL  | 0.827     | 1.00       | 0.980      |  0.871     | 1.00       |

## Top-Down Inference 

### Initial Model
|  Game               |  AP       | AP .5      | AP .75     |  AR        | AR .5      |
|---------------------|-----------|------------|------------|------------|------------|
| Test Set            | 0.684     | 0.969      | 0.781      |  0.770     | 0.983      |
| 04/11/2017 CBJ-TBL  | 0.782     | 0.980      | 0.905      |  0.817     | 0.985      |
| 08/11/2017 TBL-SJS  | 0.773     | 0.999      | 0.957      |  0.806     | 1.00       |
| 24/11/2017 TBL-WSH  | 0.527     | 0.897      | 0.589      |  0.619     | 0.949      |
| 08/11/2017 LAK-DET  | 0.736     | 0.993      | 0.842      |  0.797     | 1.00       |
| 01/12/2017 CAR-NYR  | 0.794     | 0.968      | 0.857      |  0.824     | 0.973      |
| 05/12/2017 ANA-VGK  | 0.781     | 0.996      | 0.934      |  0.831     | 1.00       |
| 07/01/2018 FLA-CBJ  | 0.730     | 1.00       | 0.896      |  0.786     | 1.00       |
| 17/01/2018 MTL-BOS  | 0.733     | 0.964      | 0.921      |  0.777     | 0.977      |
### Fine-Tuned Model

### Fine-Tuned Model
|  Game               |  AP       | AP .5      | AP .75     |  AR        | AR .5      |
|---------------------|-----------|------------|------------|------------|------------|
| Test Set            | 0.684     | 0.969      | 0.781      |  0.770     | 0.983      |
| 04/11/2017 CBJ-TBL  | 0.667     | 0.970      | 0.729      |  0.695     | 0.979      |
| 08/11/2017 TBL-SJS  | 0.790     | 0.990      | 0.938      |  0.836     | 0.995      |
| 24/11/2017 TBL-WSH  | 0.725     | 0.951      | 0.654      |  0.658     | 0.956      |
| 08/11/2017 LAK-DET  | 0.585     | 0.980      | 0.600      |  0.620     | 0.986      |
| 01/12/2017 CAR-NYR  | 0.594     | 0.967      | 0.697      |  0.629     | 0.974      |
| 05/12/2017 ANA-VGK  | 0.638     | 0.950      | 0.810      |  0.668     | 0.950      |
| 07/01/2018 FLA-CBJ  | 0.804     | 1.00       | 0.908      |  0.836     | 1.00       |
| 17/01/2018 MTL-BOS  | 0.605     | 0.980      | 0.580      |  0.649     | 0.988      |

