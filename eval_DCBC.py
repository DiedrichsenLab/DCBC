#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Aug 17 11:31:32 2020

Distance-Controlled Boundaries Coefficient (DCBC) evaluation
for a functional parcellation of brain cortex

INPUTS:
sn:                   The return subject number
hems:                 Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
binWidth:             The spatial binning width in mm, default 1 mm
maxDist:              The maximum distance for vertices pairs
parcels:              The cortical parcellation labels (integer value) to be evaluated, shape is (N,)
                      N is the number of vertices, 0 - medial wall
condType:             The condition type for evaluating
                      'unique' - evaluation will be done by using unique task conditions of the task set
                      'all' - evaluation will be done by all task conditions of the task set
taskSet:              The task set of MDTB to use for evaluating. 1 - taskset A; 2 - taskset B; [1,2] - both
resolution:           The resolution of surface space, either 32k or 164k, 32k as default
distType:             The distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                      Euclidean distance. Dijkstra's distance as default
icoRes:               Icosahedron resolution, 42, 162, 362, 642, 1002, ... default to use 2562
mwallFile:            The medial wall to be excluded from the evaluation


OUTPUT:
M:                    Gifti object- can be saved as a *.func.gii or *.label.gii file

Author: Da Zhi
'''

import os
import numpy as np
import scipy.io as spio
import scipy
from scipy.sparse import find
import nibabel as nb
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def delete_cols_csr(mat, indices):
    """
    Remove the cols denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[1], dtype=bool)
    mask[indices] = False
    return mat[:, mask]


def scan_subdirs(path):
    """Yield directory names not starting with '.' under given path."""
    subDirs = []
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            subDirs.append(entry.name)

    return subDirs


def load_subjectData(path, hemis='L'):
    """
        Load subject data in .mat and func.gii format.
        This function can be further extended depending on user's data file type, default subject data type is func.gii.

        If type = func.gii, the file name must be subNum.hemisphere.wbeta.resolution.func.gii
                            i.g 's02.L.wbeta.32k.func.gii'

        :param path: the current subject data folder path
        :return: subject data, expect data shape [N, k]
                    N is the number of vertices
                    k is the number of task conditions
    """
    files = os.listdir(path)

    if not any(".mat" or ".func.gii" in x for x in files):
        raise Exception('Input data file type is not supported.')
    elif any(".mat" in x for x in files):
        data = spio.loadmat(os.path.join(path, "file_%s.txt" % hemis))
    else:
        '''Default data type is func.gii'''
        sub = '.%s.' % hemis
        fileName = [s for s in files if sub in s]
        mat = nb.load(os.path.join(path, fileName[0]))
        wbeta_data = [x.data for x in mat.darrays]
        wbeta = np.reshape(wbeta_data, (len(wbeta_data), len(wbeta_data[0])))
        data = wbeta.transpose()

    return data


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
        mean = data.mean(axis=1)
        data = data - mean[:, np.newaxis]  # mean centering
    else:
        data = data

    # specify the condition index used to compute correlation, otherwise use all conditions
    if cond is not 'all':
        data = data[:, cond]
    elif cond is 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    k = data.shape[1]
    sd = np.sqrt(np.sum(np.square(data), axis=1) / k)  # standard deviation
    sd = np.reshape(sd, (sd.shape[0], 1))
    var = np.matmul(sd, sd.transpose())
    cov = np.matmul(data, data.transpose()) / k
    return cov, var


def compute_corr(data, cond='all', kernel='Pearson', mean_centering=True):
    """
        Compute the affinity matrix by given kernel type,
        default to calculate Pearson's correlation between all vertex pairs

        :param data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        :param cond: specify the subset of activation conditions to evaluation
                    (e.g condition column [1,2,3,4]),
                     if not given, default to use all conditions
        :param kernel: the kernel type used to calculate the affinity matrix,
                       default is Pearson's correlation
        :param mean_centering: boolean value to determine whether the given subject data
                               should be mean centered

        :return: R - the affinity matrix of current subject data. [N * N]
    """

    # mean centering of the subject beta weights
    if mean_centering:
        mean = data.mean(axis=1)
        data = data - mean[:, np.newaxis]  # mean centering
    else:
        data = data

    # specify the condition index used to compute correlation, otherwise use all conditions
    if cond is not 'all':
        data = data[:, cond]
    elif cond is 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    # Choose appropriate kernel type, default = pearson's correlation
    if kernel == 'Pearson':
        R = np.corrcoef(data)
    elif kernel == 'cosine':
        R = cosine_similarity(data)
    elif kernel == 'eclidean':
        R = 1 - euclidean_distances(data)
    else:
        raise Exception('The kernel type is not supported.')

    return R


class DCBC:
    def __init__(self, hems='all', maxDist=35, binWidth=1, parcellation=np.empty([]),
                 distType='Sphere', weighting=True):
        """
        Constructor of DCBC class
        :param hems:        Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
        :param maxDist:     The maximum distance for vertices pairs
        :param binWidth:    The spatial binning width in mm, default 1 mm
        :param parcellation:
        :param distType:    The distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                            Euclidean distance. Dijkstra's distance as default
        :param weighting:   Boolean value. True - add weighting scheme to DCBC (default)
                                           False - no weighting scheme to DCBC
        """
        self.hems = hems
        self.maxDist = maxDist
        self.binWidth = binWidth
        self.parcellation = parcellation
        self.distType = distType
        self.weighting = weighting

    def evaluate(self, parcellation):
        """
        The public function that handle the main DCBC evaluation routine

        :param parcellation: The cortical parcellation to evaluate
        :return: dict T that contain all needed DCBC evaluation results
        """
        numBins = int(np.floor(self.maxDist / self.binWidth))
        subjectsDir = scan_subdirs('data')

        if self.distType is 'Dijkstra':
            dist = spio.loadmat("distanceMatrix/distAvrg_sp.mat")['avrgDs']
            dist = dist.astype('float16')
            dist = dist.tocsr()
        elif self.distType is 'Sphere':
            dist = spio.loadmat("distanceMatrix/distSphere_sp.mat")['avrgDs']
            dist = dist.astype('float16')
            dist = dist.tocsr()
        else:
            raise TypeError("Distance type cannot be recognized!")

        # Determine which hemisphere shall be evaluated
        if self.hems is 'all':
            hems = ['L', 'R']
        elif self.hems is 'L' or 'R':
            hems = [self.hems]
        else:
            raise TypeError("Hemisphere type cannot be recognized!")

        D = dict()
        for h in hems:
            for dir in subjectsDir:
                print('evaluating %s %s ...' % (dir, h))
                path = os.path.join('data', dir)
                data = load_subjectData(path, hemis=h)

                # remove nan value and medial wall from subject data
                nanIdx = np.union1d(np.unique(np.where(np.isnan(data))[0]), np.where(parcellation == 0)[0])
                data = np.delete(data, nanIdx, axis=0)
                cov, var = compute_var_cov(data)  # This line can be changed to use compute_corr()

                # remove the nan value and medial wall from dist file
                this_dist = delete_rows_csr(dist, nanIdx)
                this_dist = delete_cols_csr(this_dist, nanIdx)
                row, col, distance = find(this_dist)

                # making parcellation matrix without medial wall and nan value
                par = np.delete(parcellation, nanIdx, axis=0)
                num_within, num_between, corr_within, corr_between = [], [], [], []
                for i in range(numBins):
                    inBin = np.where((distance > i * self.binWidth) & (distance <= (i + 1) * self.binWidth))[0]

                    # lookup the row/col index of within and between vertices
                    within = np.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
                    between = np.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

                    # retrieve and append the number of vertices for within/between in current bin
                    num_within = np.append(num_within, within.shape[0])
                    num_between = np.append(num_between, between.shape[0])

                    # Compute and append averaged within- and between-parcel correlations in current bin
                    this_corr_within = np.nanmean(cov[row[inBin[within]], col[inBin[within]]]) / np.nanmean(var[row[inBin[within]], col[inBin[within]]])
                    this_corr_between = np.nanmean(cov[row[inBin[between]], col[inBin[between]]]) / np.nanmean(var[row[inBin[between]], col[inBin[between]]])
                    corr_within = np.append(corr_within, this_corr_within)
                    corr_between = np.append(corr_between, this_corr_between)

                    del inBin

                if self.weighting:
                    weight = 1/(1/num_within + 1/num_between)
                    weight = weight / np.sum(weight)
                    DCBC = np.nansum(np.multiply((corr_within - corr_between), weight))
                else:
                    DCBC = np.nansum(corr_within - corr_between)
                    weight = np.nan

                D[dir + '_' + h] = {
                    "binWidth": self.binWidth,
                    "maxDist": self.maxDist,
                    "hemisphere": h,
                    "num_within": num_within,
                    "num_between": num_between,
                    "corr_within": corr_within,
                    "corr_between": corr_between,
                    "weight": weight,
                    "DCBC": DCBC
                }

        return D



