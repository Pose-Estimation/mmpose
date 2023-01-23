import os
import torch
import pickle
import numpy as np

import inteutil
from matching import posematcher
import networkinte

from tqdm import tqdm
from TorchSUL import Model as M
from collections import defaultdict

if __name__ == '__main__':
    ## step 1: match the poses
    print('Matching poses from two branches...')
    matcher = posematcher.PoseMatcher(
        top_down_path='./integration/td_keypoints.json',
        btm_up_path='./integration/bu_keypoints.json')
    matcher.match(pts_out_path='./integration/pred_bu/')

    # step 2: infer the integrated results
    print('Inferring the integrated poses...')
    # create data loader
    data = inteutil.InteDataset(
        bu_path='./mupots/pred_bu/', td_path='./mupots/pred/')
    # initialize the network
    net = networkinte.IntegrationNet()
    pts_dumb = torch.zeros(2, 84)
    net(pts_dumb)
    M.Saver(net).restore('./ckpts/model_inte/')
    net.cuda()

    # create paths
    if not os.path.exists('./mupots/pred_inte/'):
        os.makedirs('./mupots/pred_inte/')

    with torch.no_grad():
        all_pts = defaultdict(list)
        for src_pts, src_dep, vid_inst in tqdm(data):
            src_pts = torch.from_numpy(src_pts).cuda()
            res_pts = net(src_pts)
            res_pts = res_pts.cpu().numpy()

            # save results
            # TODO change to coco format??
            i, j = vid_inst
            all_pts[i].insert(j, res_pts)

        for k in all_pts:
            result = np.stack(all_pts[k], axis=1)
            pickle.dump(result,
                        open('./mupots/pred_inte/%d.pkl' % (k + 1), 'wb'))
