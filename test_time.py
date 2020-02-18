#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import argparse
import numpy as np
import os
from tqdm import tqdm
from data import ModelNet40
import time
import types
import torch
from torch.utils.data import DataLoader
from model import IDAM, FPFH, GNN

from eval_open3d_models import ICPModel, FGRModel
from eval_dcp_model import DCPModel
from eval_pointnetlk_model import PointNetLKModel

from prnet.model import PRNet

class PRnetModel():
    def __init__(self):
        args = types.SimpleNamespace()
        args.n_iters = 3
        args.discount_factor = 0.9
        args.feature_alignment_loss = 0.1
        args.cycle_consistency_loss = 0.1
        model_path = 'prnet/checkpoints/ss/models/model.best.t7'
        args.n_emb_dims = 512
        args.n_keypoints = 512
        args.n_subsampled_points = 512
        args.emb_nn = 'dgcnn'
        args.attention = 'transformer'
        args.head = 'svd'
        args.n_blocks = 1
        args.n_heads = 4
        args.dropout = 0.0
        args.n_ff_dims = 1024
        args.temp_factor = 100
        args.cat_sampler = 'gumbel_softmax'
        self.net = PRNet(args).cuda()
        self.net.load_state_dict(torch.load(model_path), strict=False)
        self.net.eval()

    def __call__(self, src, tgt):
        src = torch.from_numpy(src).float().cuda()
        tgt = torch.from_numpy(tgt).float().cuda()
        R, t, *_ = self.net(src, tgt)
        R = R.squeeze(0).detach().cpu().numpy()
        t = t.squeeze(0).detach().cpu().numpy()
        return R, t





class IDAM_FPFH():
    def __init__(self):
        args = types.SimpleNamespace()
        args.emb_dims = 33
        args.num_iter = 3
        self.net = IDAM(FPFH(), args).cuda()
        model_path = 'checkpoints/exp_FPFH/models/model.4.t7'
        self.net.load_state_dict(torch.load(model_path), strict=True)
        self.net.eval()

    def __call__(self, src, tgt):
        src = torch.from_numpy(src).float().cuda()
        tgt = torch.from_numpy(tgt).float().cuda()
        R, t, _ = self.net(src, tgt)
        R = R.cpu().numpy()
        t = t.cpu().numpy()
        return R, t


class IDAM_GNN():
    def __init__(self):
        args = types.SimpleNamespace()
        args.emb_dims = 64
        args.num_iter = 3
        self.net = IDAM(GNN(), args).cuda()
        model_path = 'checkpoints/exp/models/model.19.t7'
        self.net.load_state_dict(torch.load(model_path), strict=True)
        self.net.eval()

    def __call__(self, src, tgt):
        print('debug')
        start = time.time()
        src = torch.from_numpy(src).float().cuda()
        tgt = torch.from_numpy(tgt).float().cuda()
        print('a', time.time()-start)
        start = time.time()
        R, t, _ = self.net(src, tgt)
        print('b', time.time()-start)
        start = time.time()
        R = R.cpu().numpy()
        t = t.cpu().numpy()
        print('c', time.time()-start)
        return R, t


def test_time(net, test_data):
    time_list = []

    for src, tgt, *_ in tqdm(test_data):

        # the channel order returned should be consistent with pytorch
        # and the type should all be numpy arrays
        src = src.numpy()
        tgt = tgt.numpy()
        start = time.time()
        rotation_ab_pred, translation_ab_pred = net(src, tgt)
        # torch.cuda.synchronize()
        time_list.append((time.time()-start)/src.shape[0])

    mean_time = np.mean(time_list)
    print('This model takes %.5f seconds per point cloud' % mean_time)


if __name__ == '__main__':
    ####### Things to configure ######
    num_points = 1024
    factor = 4
    batch_size = 16
    # model = DCPModel('dcp/checkpoints/wwi/models/model.best.t7')
    # model = PointNetLKModel('PointNetLK/experiments/our_exp/results/ex1_pointlk_0915_wwi_model_last.pth')
    # model = PRnetModel()
    # model = IDAM_FPFH()
    # model = ICPModel()
    model = IDAM_GNN()
    ####### Things to configure ######


    dataset = ModelNet40(num_points=num_points, partition='test', gaussian_noise=False, alpha=1., unseen=True, factor=4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_time(model, loader)
