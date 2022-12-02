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
| 04/11/2017 CBJ-TBL  | 0.493     | 0.768      | 0.611      |  0.526     | 0.779      |
| 08/11/2017 TBL-SJS  | 0.677     | 0.971      | 0.908      |  0.718     | 0.981      |
| 24/11/2017 TBL-WSH  | 0.635     | 0.955      | 0.762      |  0.674     | 0.962      |
| 08/11/2017 LAK-DET  | 0.507     | 0.859      | 0.538      |  0.570     | 0.871      |
| 01/12/2017 CAR-NYR  | 0.621     | 0.946      | 0.774      |  0.671     | 0.957      |
| 05/12/2017 ANA-VGK  | 0.413     | 0.689      | 0.507      |  0.435     | 0.709      |
| 07/01/2018 FLA-CBJ  | 0.746     | 1.00       | 0.944      |  0.769     | 1.00       |
| 17/01/2018 MTL-BOS  | 0.525     | 0.780      | 0.581      |  0.570     | 0.792      |

### Fine Tuned Model
|  Game               |  AP       | AP .5      | AP .75     |  AR        | AR .5      |
|---------------------|-----------|------------|------------|------------|------------|
| Test Set            | 0.684     | 0.969      | 0.781      |  0.770     | 0.983      |
| 04/11/2017 CBJ-TBL  | 0.704     | 0.955      | 0.765      |  0.770     | 0.971      |
| 08/11/2017 TBL-SJS  | 0.841     | 0.986      | 0.966      |  0.878     | 0.995      |
| 24/11/2017 TBL-WSH  | 0.685     | 0.972      | 0.854      |  0.768     | 0.984      |
| 08/11/2017 LAK-DET  | 0.623     | 0.980      | 0.689      |  0.702     | 0.990      |
| 01/12/2017 CAR-NYR  | 0.634     | 0.968      | 0.730      |  0.672     | 0.974      |
| 05/12/2017 ANA-VGK  | 0.684     | 0.940      | 0.847      |  0.721     | 0.943      |
| 07/01/2018 FLA-CBJ  | 0.819     | 1.00       | 0.913      |  0.860     | 1.00       |
| 17/01/2018 MTL-BOS  | 0.675     | 1.00       | 0.657      |  0.765     | 1.00       |

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

