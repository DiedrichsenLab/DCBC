#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 25/09/2024
Pytorch version of dcbc calculation for speed up.

Author: Barafat
'''

import numpy as np
import torch as pt


def compute_var_cov(data, cond='all', mean_centering=True):
    """
        Compute the affinity matrix by given kernel type,
        default to calculate Pearson's correlation between all vertex pairs

        :param data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        :param cond: specify the subset of activation conditions to evaluation
                    (e.g condition column [1,2,3,4]),
                     if not given, default to use all conditions
        :param mean_centering: boolean value to determine whether the given subject data
                               should be mean centered

        :return: cov - the covariance matrix of current subject data. shape [N * N]
                 var - the variance matrix of current subject data. shape [N * N]
    """
    if mean_centering:
        data = data - pt.mean(data, dim=1, keepdim=True)  # mean centering

    # specify the condition index used to compute correlation, otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    k = data.shape[1]
    cov = pt.matmul(data, data.T) / (k - 1)
    # sd = data.std(dim=1).reshape(-1, 1)  # standard deviation
    sd = pt.sqrt(pt.sum(data ** 2, dim=1, keepdim=True) / (k - 1))
    var = pt.matmul(sd, sd.T)

    return cov, var


def compute_dist(coord, resolution=2):
    """
    calculate the distance matrix between each of the voxel pairs by given mask file

    :param coord: the ndarray of all N voxels coordinates x,y,z. Shape N * 3
    :param resolution: the resolution of .nii file. Default 2*2*2 mm

    :return: a distance matrix of N * N, where N represents the number of masked voxels
    """
    if type(coord) is np.ndarray:
        coord = pt.tensor(coord, dtype=pt.get_default_dtype())

    num_points = coord.shape[0]
    D = pt.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return pt.sqrt(D) * resolution


def compute_DCBC(maxDist=35, binWidth=1, parcellation=np.empty([]),
                 func=None, dist=None, weighting=True):
    """
    The main entry of DCBC calculation for volume space
    :param hems:        Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
    :param maxDist:     The maximum distance for vertices pairs
    :param binWidth:    The spatial binning width in mm, default 1 mm
    :param parcellation:
    :param dist_file:   The path of distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                        Euclidean distance. Dijkstra's distance as default
    :param weighting:   Boolean value. True - add weighting scheme to DCBC (default)
                                       False - no weighting scheme to DCBC
    """
    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func)
    # cor = np.corrcoef(func)
    if not dist.is_sparse:
        dist = dist.to_sparse()

    row = dist._indices()[0]
    col = dist._indices()[1]
    distance = dist._values()
    # row, col, distance = sp.sparse.find(dist)

    # making parcellation matrix without medial wall and nan value
    par = parcellation
    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = pt.where((distance > i * binWidth) &
                         (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = pt.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
        between = pt.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within.append(
            pt.tensor(within.numel(), dtype=pt.get_default_dtype()))
        num_between.append(
            pt.tensor(between.numel(), dtype=pt.get_default_dtype()))

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr_within = pt.nanmean(cov[row[inBin[within]], col[inBin[within]]]) \
            / pt.nanmean(var[row[inBin[within]], col[inBin[within]]])
        this_corr_between = pt.nanmean(cov[row[inBin[between]], col[inBin[between]]]) \
            / pt.nanmean(var[row[inBin[between]], col[inBin[between]]])

        corr_within.append(this_corr_within)
        corr_between.append(this_corr_between)

        del inBin

    if weighting:
        weight = 1 / (1 / pt.stack(num_within) + 1 / pt.stack(num_between))
        weight = weight / pt.sum(weight)
        DCBC = pt.nansum(pt.multiply(
            (pt.stack(corr_within) - pt.stack(corr_between)), weight))
    else:
        DCBC = pt.nansum(pt.stack(corr_within) - pt.stack(corr_between))
        weight = pt.nan

    D = {
        "binWidth": binWidth,
        "maxDist": maxDist,
        "num_within": num_within,
        "num_between": num_between,
        "corr_within": corr_within,
        "corr_between": corr_between,
        "weight": weight,
        "DCBC": DCBC
    }

    return D