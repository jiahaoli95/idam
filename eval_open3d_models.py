#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import open3d as o3d
import numpy as np


def npy2pcd(npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    return pcd


def FPFH(xyz):
    # set voxel_size
    voxel_size = 0.05
    # xyz is numpy array
    pcd = npy2pcd(xyz)
    # downsample
    # pcd_down = o3d.voxel_down_sample(pcd, voxel_size)
    # estimate normals
    radius_normal = voxel_size * 2
    #radius_normal = 0.05
    o3d.estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # compute fpfh
    radius_feature = voxel_size * 5
    #radius_feature = 0.05
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd, pcd_fpfh


class FGRModel():
    def __init__(self):
        pass

    def __call__(self, src, tgt):
        R = []
        t = []
        for i in range(src.shape[0]):
            sc = src[i].transpose()
            tt = tgt[i].transpose()
            sc, src_fpfh = FPFH(sc)
            tt, tgt_fpfh = FPFH(tt)
            result = o3d.registration.registration_fast_based_on_feature_matching(
                sc, tt, src_fpfh, tgt_fpfh,
                o3d.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=0.02))
            result = result.transformation
            R.append(result[:3, :3])
            t.append(result[:3, 3])
        return R, t


class ICPModel():
    def __init__(self, align_center=False):
        self.trans_init = np.asarray([[1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 1., 0.],
                                      [0., 0., 0., 1.]])
        self.threshold = 100
        self.align_center = align_center

    def __call__(self, src, tgt):
        if self.align_center:
            t_init = tgt.mean(-1) - src.mean(-1)
            trans_init = self.trans_init.copy()
            trans_init[:3, 3] = t_init
        else:
            trans_init = self.trans_init.copy()
        R = []
        t = []
        for i in range(src.shape[0]):
            sc = src[i].transpose()
            tt = tgt[i].transpose()
            sc = npy2pcd(sc)
            tt = npy2pcd(tt)
            reg_p2p = o3d.registration.registration_icp(
                sc, tt, self.threshold, trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-19, relative_rmse=1e-19, max_iteration=500))
            reg_p2p = reg_p2p.transformation
            R.append(reg_p2p[:3, :3])
            t.append(reg_p2p[:3, 3])
        return R, t


if __name__ == '__main__':
    pass
