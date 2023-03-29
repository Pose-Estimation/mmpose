import argparse
from collections import defaultdict
import json
import numpy as np
from integration.matching import posematcher
from integration.matching.utils import format_annotations
from mmpose.core.evaluation.top_down_eval import keypoint_pck_accuracy
import torch
from tqdm import tqdm
import TorchSUL.Model as M
from matching.utils import format_keypoints_mask

from test_dataset import TestInteDataset
from networkinte import IntegrationNet

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
    parser.add_argument('td_path', help='Path to the top-down results')
    parser.add_argument('bu_path', help='Path to the bottom-up results')
    parser.add_argument(
        'inte_path', help='Path to integration network checkpoint')
    parser.add_argument('gt_path', help='Path to ground truth annotations')
    # parser.add_argument(
    #     '--gpu-id',
    #     type=int,
    #     default=0,
    #     help='id of gpu to use '
    #     '(only applicable to non-distributed testing)')
    args = parser.parse_args()
    return args


def normalize_kpt(kpt):
    norm_kpt = kpt[0, :] / 640
    norm_kpt = norm_kpt[1, :] / 360
    return norm_kpt


# def load_model(cfg, checkpoint):
#     model = build_posenet(cfg.model)
#     fp16_cfg = cfg.get('fp16', None)
#     if fp16_cfg is not None:
#         wrap_fp16_model(model)
#     load_checkpoint(model, checkpoint, map_location='cpu')

#     model = MMDataParallel(model, device_ids=[args.gpu_id])
#     return model


def format_pck(keypoints):

    coco_keypoints = np.zeros((1, 14, 2))
    for i, coordinates in enumerate(keypoints):
        for j, point in enumerate(coordinates):
            coco_keypoints[0, j, i % 2] = point
    return coco_keypoints


if __name__ == "__main__":

    args = arg_parser()

    PATH_TO_TOP_DOWN = args.td_path
    PATH_TO_BOTTOM_UP = args.bu_path
    PATH_TO_GROUND_TRUTH = args.gt_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    bottom_up_kpts = np.array(matcher.match())
    bu_formatted = format_annotations(PATH_TO_BOTTOM_UP)

    # Getting td annotations
    f = open(PATH_TO_TOP_DOWN)
    top_down_data = json.load(f)
    f.close()
    td_formatted = format_annotations(PATH_TO_TOP_DOWN)

    test_loader = TestInteDataset(bottom_up_kpts, top_down_data)

    # Initialize/load the network
    net = IntegrationNet()
    pts_dumb = torch.zeros(2, 84)
    net(pts_dumb)

    PATH_TO_TD = args.inte_path
    M.Saver(net).restore(PATH_TO_TD)
    net.to(device)

    all_pts = defaultdict(list)
    with torch.no_grad():
        net.eval()

        for src_pts, img_id in tqdm(test_loader):
            res_pts = net(src_pts)
            res_pts = res_pts.cpu().numpy()
            res_pts = res_pts.reshape(3, 14)
            res_pts[0, :] = res_pts[0, :] * 640
            res_pts[1, :] = res_pts[1, :] * 360
            res_pts = res_pts[0:2]
            # save results
            all_pts[img_id].append(res_pts)

    # Open ground truth annotations
    f = open(PATH_TO_GROUND_TRUTH)
    ground_truth_data = json.load(f)
    f.close()
    ground_truth_annots = ground_truth_data["annotations"]

    pck = pck_td = pck_bu = 0
    pck_total = 0
    prev = None
    counter = 0
    normalize = np.ones((1, 2))
    normalize[0, :] = (640, 360)

    log_file = open("./integration/pck.md", "w")
    log_file.write(
        "| Image | Pck Integration | PCK Bottom Up| Pck Top Down\n|:-----:|:----------:|:----------:|:----------:|\n"
    )

    for pose in ground_truth_annots:
        gt_keypoints, mask = format_keypoints_mask(pose["keypoints"])
        img_id = pose["image_id"]

        # Calculate PCK
        pred = all_pts[img_id]

        match, oks = matcher._best_match(gt_keypoints, pred)
        if isinstance(match, np.ndarray):
            bu_match = format_pck(
                matcher._best_match(gt_keypoints, bu_formatted[img_id])[0])
            td_match = format_pck(
                matcher._best_match(gt_keypoints, td_formatted[img_id])[0])

            match = format_pck(match)
            gt_keypoints = format_pck(gt_keypoints)

            mask = mask.reshape(1, 14)
            pck += keypoint_pck_accuracy(match, gt_keypoints, mask, 0.1,
                                         normalize)[1]

            #Compare with initial pck for top down
            pck_td += keypoint_pck_accuracy(td_match, gt_keypoints, mask, 0.1,
                                            normalize)[1]
            #Compare with initial pck for bottom-up
            pck_bu += keypoint_pck_accuracy(bu_match, gt_keypoints, mask, 0.1,
                                            normalize)[1]

            if prev is not None and prev != img_id:

                pck_total += pck
                log_file.write(
                    f'|{prev}|{pck/counter:.4}|{pck_bu/counter:.4} |{pck_td/counter:.4} |\n'
                )
                # print("-" * 80)
                # print(f'Average PCK for image: {prev} is {pck/counter:.4}')
                # print("-" * 80)
                pck = pck_td = pck_bu = counter = 0

        counter += 1
        prev = img_id

    print("-" * 80)
    print(f'Overall PCK {pck_total/len(ground_truth_annots)}')
    print("-" * 80)

    log_file.close()
