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


class IDAM_FPFH():
    def __init__(self):
        args = types.SimpleNamespace()
        args.emb_dims = 33
        args.num_iter = 3
        self.net = IDAM(FPFH(), args).cuda()
        model_path = 'checkpoints/exp/models/model.25.t7'
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
        model_path = 'checkpoints/exp/models/model.0.t7'
        self.net.load_state_dict(torch.load(model_path), strict=True)
        self.net.eval()

    def __call__(self, src, tgt):
        src = torch.from_numpy(src).float().cuda()
        tgt = torch.from_numpy(tgt).float().cuda()
        R, t, _ = self.net(src, tgt)
        R = R.cpu().numpy()
        t = t.cpu().numpy()
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
        torch.cuda.synchronize()
        time_list.append((time.time()-start)/src.shape[0])

    mean_time = np.mean(time_list)
    print('This model takes %.5f seconds per point cloud' % mean_time)


if __name__ == '__main__':
    ####### Things to configure ######
    num_points = 2048
    factor = 4
    batch_size = 16
    model = IDAM_GNN()
    ####### Things to configure ######


    dataset = ModelNet40(num_points=num_points, partition='test', gaussian_noise=False, alpha=1., unseen=True, factor=4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_time(model, loader)
