#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 10/6/2021
The DCBC evaluation function for volume space parcellations

Author: DZHI
'''
import numpy as np
import scipy as sp
from DCBC.utilities import compute_var_cov

# Check if torch is available
try:
    import torch as pt
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_DCBC(maxDist=35, binWidth=1, parcellation=np.empty([]),
                 func=None, dist=None, weighting=True, backend='torch'):
    """ DCBC calculation.
        Automatically chooses the backend or uses user-specified backend.

    Args:
        maxDist: The maximum distance for vertices pairs, default 35 mm
        binWidth: The spatial binning width in mm, default 1 mm
        parcellation: A 1-d vector tensor of brain parcellation to be
                      evaluated
        func: the functional data for evaluating, shape (N, P),
              N - the dimensionality of underlying data, i.e. the number
              of task contrasts or the number of resting-state networks
              P - the number of brain voxels / vertices
        dist: the pairwise distance matrix between P brain locations. It
              can be a dense matrix or sparse tensor
        weighting: If True, the DCBC result is weighted averaged across
                   spatial bins. If False, it is plain averaged.
        backend: the backend for the calculation. If "numpy", then following
                 calculation will be using numpy. If "torch", then following
                 calculation will be on PyTorch.

    Returns:
        D: a dictionary contains necessary information for DCBC analysis
    """
    if backend == 'torch' and TORCH_AVAILABLE:
        if type(parcellation) is np.ndarray:
            parcellation = pt.tensor(parcellation, dtype=pt.get_default_dtype())
        if type(func) is np.ndarray:
            func = pt.tensor(func, dtype=pt.get_default_dtype())
        assert all(pt.is_tensor(v) for v in [parcellation, func, dist]),\
            "All inputs must be pytorch tensors!"
        return compute_DCBC_pt(maxDist=maxDist, binWidth=binWidth,
                               parcellation=parcellation, func=func,
                               dist=dist, weighting=weighting)
    elif backend == 'numpy' or not TORCH_AVAILABLE:
        return compute_DCBC_np(maxDist=maxDist, binWidth=binWidth,
                               parcellation=parcellation, func=func,
                               dist=dist, weighting=weighting)
    else:
        raise ValueError("Torch not available and no valid backend specified!")


def compute_DCBC_np(maxDist=35, binWidth=1, parcellation=np.empty([]),
                    func=None, dist=None, weighting=True):
    """ DCBC calculation (Numpy version)

    Args:
        maxDist: The maximum distance for vertices pairs, default 35 mm
        binWidth: The spatial binning width in mm, default 1 mm
        parcellation: A 1-d vector tensor of brain parcellation to be
                      evaluated
        func: the functional data for evaluating, shape (N, P),
              N - the dimensionality of underlying data, i.e. the number
              of task contrasts or the number of resting-state networks
              P - the number of brain voxels / vertices
        dist: the pairwise distance matrix between P brain locations. It
              can be a dense matrix or sparse tensor
        weighting: If True, the DCBC result is weighted averaged across
                   spatial bins. If False, it is plain averaged.

    Returns:
        D: a dictionary contains necessary information for DCBC analysis
    """
    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func, backend='numpy')

    # remove the nan value and medial wall from dist file
    row, col, distance = sp.sparse.find(dist)

    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = np.where((distance > i * binWidth) & (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = np.where((parcellation[row[inBin]] == parcellation[col[inBin]]) == True)[0]
        between = np.where((parcellation[row[inBin]] == parcellation[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within = np.append(num_within, within.shape[0])
        num_between = np.append(num_between, between.shape[0])

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr_within = np.nanmean(cov[row[inBin[within]], col[inBin[within]]]) \
                           / np.nanmean(var[row[inBin[within]], col[inBin[within]]])
        this_corr_between = np.nanmean(cov[row[inBin[between]], col[inBin[between]]]) \
                            / np.nanmean(var[row[inBin[between]], col[inBin[between]]])

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


def compute_DCBC_pt(maxDist=35, binWidth=1, parcellation=np.empty([]),
                 func=None, dist=None, weighting=True):
    """ DCBC calculation (PyTorch version)

    Args:
        maxDist: The maximum distance for vertices pairs, default 35 mm
        binWidth: The spatial binning width in mm, default 1 mm
        parcellation: A 1-d vector tensor of brain parcellation to be
                      evaluated
        func: the functional data for evaluating, shape (N, P),
              N - the dimensionality of underlying data, i.e. the number
              of task contrasts or the number of resting-state networks
              P - the number of brain voxels / vertices
        dist: the pairwise distance matrix between P brain locations. It
              can be a dense matrix or sparse tensor
        weighting: If True, the DCBC result is weighted averaged across
                   spatial bins. If False, it is plain averaged.

    Returns:
        D: a dictionary contains necessary information for DCBC analysis
    """
    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func, backend='torch')
    # cor = np.corrcoef(func)
    if not dist.is_sparse:
        dist = dist.to_sparse()
    row, col = dist._indices()
    distance = dist._values()

    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = pt.where((distance > i * binWidth) &
                         (distance <= (i + 1) * binWidth))[0]
        # lookup the row/col index of within and between vertices
        within = pt.where((parcellation[row[inBin]] == parcellation[col[inBin]]) == True)[0]
        between = pt.where((parcellation[row[inBin]] == parcellation[col[inBin]]) == False)[0]
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

    D = {"binWidth": binWidth,
         "maxDist": maxDist,
         "num_within": num_within,
         "num_between": num_between,
         "corr_within": corr_within,
         "corr_between": corr_between,
         "weight": weight,
         "DCBC": DCBC}

    return D