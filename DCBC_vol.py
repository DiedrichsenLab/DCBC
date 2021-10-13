#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 10/6/2021
The DCBC evaluation function for volume space parcellations

Author: DZHI
'''
import numpy as np
import scipy as sp
from eval_DCBC import compute_var_cov
import nibabel as nb
import mat73
from plotting import plot_single


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


def compute_dist(coord, resolution=2):
    """
    calculate the distance matrix between each of the voxel pairs by given mask file

    :param coord: the ndarray of all N voxels coordinates x,y,z. Shape N * 3
    :param resolution: the resolution of .nii file. Default 2*2*2 mm

    :return: a distance matrix of N * N, where N represents the number of masked voxels
    """

    num_points = coord.shape[0]
    D = np.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return np.sqrt(D) * resolution


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

    # remove the nan value and medial wall from dist file
    row, col, distance = sp.sparse.find(dist)

    # making parcellation matrix without medial wall and nan value
    par = parcellation
    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = np.where((distance > i * binWidth) & (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = np.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
        between = np.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within = np.append(num_within, within.shape[0])
        num_between = np.append(num_between, between.shape[0])

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr_within = np.nanmean(cov[row[inBin[within]], col[inBin[within]]]) / np.nanmean(var[row[inBin[within]], col[inBin[within]]])
        this_corr_between = np.nanmean(cov[row[inBin[between]], col[inBin[between]]]) / np.nanmean(var[row[inBin[between]], col[inBin[between]]])

        # this_corr_within = np.nanmean(cor[row[inBin[within]], col[inBin[within]]])
        # this_corr_between = np.nanmean(cor[row[inBin[between]], col[inBin[between]]])

        corr_within = np.append(corr_within, this_corr_within)
        corr_between = np.append(corr_between, this_corr_between)

        del inBin

    if weighting:
        weight = 1/(1/num_within + 1/num_between)
        weight = weight / np.sum(weight)
        DCBC = np.nansum(np.multiply((corr_within - corr_between), weight))
    else:
        DCBC = np.nansum(corr_within - corr_between)
        weight = np.nan

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
