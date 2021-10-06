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
from nilearn.masking import apply_mask


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


def compute_dist(mask_file, resolution=2):
    """
    calculate the distance matrix between each of the voxel pairs by given mask file
    :param mask:
    :param resolution:
    :return: a distance matrix of N * N, where N represents the number of masked voxels
    """
    mask = nb.load(mask_file).get_data()
    coord = np.nonzero(mask.astype(int))
    x, y, z = coord[0], coord[1], coord[2]
    num_vol = len(x)
    dist = np.zeros((num_vol, num_vol))
    for i in range(len(x)):
        for j in range(len(x)):
            dist[i][j] = np.sqrt((x[i]-x[j]) ^ 2 + (y[i]-y[j]) ^ 2 + (z[i]-z[j]) ^ 2)

    return dist


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


if __name__ == "__main__":
    print('Start evaluating DCBC sample code ...')

    # Load parcellation given the mask file
    parcels = apply_mask('D:/data/sc2/encoding/glm7/spect/masked_par_choi_7.nii.gz',
                         'D:/data/sc2/encoding/glm7/spect/striatum_mask_2mm.nii')

    # Compute the distance matrix between voxel pairs using the mask file
    dist = compute_dist('D:/data/sc2/encoding/glm7/spect/striatum_mask_2mm.nii', 2)


    # Load functional profile (betas) to calculate the correlations of voxel pairs
    vol_ind = mat73.loadmat('D:/data/sc2/encoding/glm7/striatum_avrgDataStruct.mat')['volIndx']
    vol_ind = vol_ind.astype(int)

    masked_data = apply_mask('D:/data/sc2/encoding/glm7/spect/masked_par_choi_7.nii.gz','D:/data/sc2/encoding/glm7/spect/striatum_mask_2mm.nii')

    T = compute_DCBC(hems='L', maxDist=35, binWidth=1).evaluate(parcels)

    plot_wb_curve(T, path='data', hems='all')
    print('Done')
