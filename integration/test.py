import argparse
from collections import defaultdict
import json
import os
import pickle
import warnings
import numpy as np
import torch
from tqdm import tqdm
import TorchSUL.Model as M
from matching.utils import format_keypoints_mask

from test_dataset import TestInteDataset
from networkinte import IntegrationNet
from matching import posematcher

# from mmcv import Config
# from mmpose.models import build_posenet
# from mmcv.runner import load_checkpoint
# from mmcv.parallel import MMDataParallel

# try:
#     from mmcv.runner import wrap_fp16_model
# except ImportError:
#     warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
#                   'Please install mmcv>=1.1.4')
#     from mmpose.core import wrap_fp16_model


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='test',
        description=
        'Run inference on top-down, bottom-up networks with integration network',
    )
    # parser.add_argument('configtd', help='test config file path top-down')
    # parser.add_argument('checkpointtd', help='checkpoint file top-down')
    # parser.add_argument('configbu', help='test config file path bottom-up')
    # parser.add_argument('checkpointbu', help='checkpoint file bottom-up')
    parser.add_argument(
        '--td-path', '-td', help='Path to the top-down results')
    parser.add_argument(
        '--bu-path', '-bu', help='Path to the bottom-up results')
    parser.add_argument(
        '--inte-path', '-inte', help='Path to integration network checkpoint')
    parser.add_argument(
        '--gt-path', '-gt', help='Path to ground truth annotations')
    # parser.add_argument(
    #     '--gpu-id',
    #     type=int,
    #     default=0,
    #     help='id of gpu to use '
    #     '(only applicable to non-distributed testing)')
    args = parser.parse_args()
    return args


# def load_model(cfg, checkpoint):
#     model = build_posenet(cfg.model)
#     fp16_cfg = cfg.get('fp16', None)
#     if fp16_cfg is not None:
#         wrap_fp16_model(model)
#     load_checkpoint(model, checkpoint, map_location='cpu')

#     model = MMDataParallel(model, device_ids=[args.gpu_id])
#     return model

if __name__ == "__main__":

    args = arg_parser()

    PATH_TO_TOP_DOWN = args.td_path
    PATH_TO_BOTTOM_UP = args.bu_path
    PATH_TO_GROUND_TRUTH = args.gt_path

    # Load top-down model
    # cfg = Config.fromfile(args.configtd)
    # td = load_model(cfg, args.checkpointd)

    # Load bottom_up
    # cfg = Config.fromfile(args.configbu)
    # bu = load_model(cfg, args.checkpointbu)

    # Match the poses
    print("Matching poses from two branches...")
    matcher = posematcher.PoseMatcher(
        top_down_path=PATH_TO_TOP_DOWN,
        btm_up_path=PATH_TO_BOTTOM_UP,
    )
    bottom_up_kpts = matcher.match(pts_out_path="./integration/pred_bu/")

    # Getting td annotations
    f = open(PATH_TO_TOP_DOWN)
    top_down_data = json.load(f)
    f.close()
    top_down_annots = top_down_data["annotations"]

    test_loader = TestInteDataset(
        bottom_up_kpts, top_down_annots, batch_size=32)

    # Initialize/load the network
    net = IntegrationNet()
    pts_dumb = torch.zeros(2, 56)
    net(pts_dumb)

    PATH_TO_TD = args.inte_path
    M.Saver(net).restore(PATH_TO_TD)
    net.cuda()

    # create paths
    if not os.path.exists("/pred_inte/"):
        os.makedirs("/pred_inte/")

    all_pts = defaultdict(list)
    with torch.no_grad():
        net.eval()

        for src_pts, img_id in tqdm(test_loader):
            src_pts = torch.from_numpy(src_pts).cuda()
            res_pts = net(src_pts)
            res_pts = res_pts.cpu().numpy()

            # save results
            all_pts[img_id].append(res_pts)

    # Calculate PCK

    # Open ground truth annotations
    f = open(PATH_TO_TOP_DOWN)
    ground_truth_data = json.load(f)
    f.close()
    ground_truth_annots = ground_truth_data["annotations"]
    ground_truth = []
    gt_img_ids = []

    for pose in ground_truth_annots:
        ground_truth.append(format_keypoints_mask(pose["keypoints"]))
        gt_img_ids.append(pose["image_id"])
        
    
