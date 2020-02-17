"""
    Place the PointNetLK repo at the same directory as this script.

"""

import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision

from PointNetLK import ptlk


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class PointNetLKModel():
    def __init__(self, ckpt):
        # modify the following configuration to be consistent with ckpt

        args = AttrDict()
        args.use_tnet = False
        args.dim_k = 1024
        args.symfn = 'max'
        args.pointnet = 'tune'
        args.transfer_from = ''
        args.delta = 1.0e-2
        args.learn_delta = False
        args.max_iter = 10
        args.device = 'cuda:0'
        args.pretrained = ''

        self.xtol = 1.0e-7
        self.max_iter = 10
        self.p0_zero_mean = True
        self.p1_zero_mean = True
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg

        ptnet = ptlk.pointnet.PointNet_features(args.dim_k, use_tnet=args.use_tnet, sym_fn=self.sym_fn)
        model = ptlk.pointlk.PointLK(ptnet, args.delta, args.learn_delta)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.to(args.device)

        self.model = model

    def __call__(self, src, tgt):
        src = torch.from_numpy(src).float().cuda()
        tgt = torch.from_numpy(tgt).float().cuda()
        src = src.transpose(1, 2)
        tgt = tgt.transpose(1, 2)
        # src = src.unsqueeze(0)
        # tgt = tgt.unsqueeze(0)
        model = self.model
        _ = ptlk.pointlk.PointLK.do_forward(model, tgt, src, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean)
        G = model.g
        R = G[:, :3, :3]
        t = G[:, :3, 3]
        R = R.squeeze(0).detach().cpu().numpy()
        t = t.squeeze(0).detach().cpu().numpy()
        return R, t
