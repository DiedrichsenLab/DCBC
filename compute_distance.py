#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Feb 25 10:16:23 2021

The function of calculating distance matrix between each pair
of vertices on the cortical surface

INPUTS:     files:      The geometrical information of the cortical surface.
                        The default file type is 'surf.gii' as the project is
                        built upon HCP fs_LR surface space. Currently not
                        support other types of geometry files
            type:       The algorithm used to calculate the distance between
                        vertex pairs
            max_dist:   The maximum distance of

OUTPUT:     The expected distance matrix using given algorithm
            shape [N, N] where N indicates the size of vertices, and the
            value at N_i and N_j is the distance of the vertices i and j

Author: Da Zhi
'''
import numpy as np
import nibabel as nb
from scipy import sparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def euclidean_distance(a, b, decimals=3):
    """
    Compute euclidean similarity between samples in a and b.
        K(X, Y) = <X, Y> / (||X||*||Y||)

        :param a: ndarray, shape: (n_samples, n_features) input data.
                  e.g (32492, 34) means 32,492 cortical nodes with 34 task
                  condition activation profile
        :param b: ndarray, shape: (n_samples, n_features) input data.
                  If None, b = a

        :return:  r - the cosine similarity matrix between nodes. [N * N]
                  N is the number of cortical nodes
    """
    p1 = np.einsum('ij,ij->i', a, a)[:, np.newaxis]
    p2 = np.einsum('ij,ij->i', b, b)[:, np.newaxis]
    p3 = -2 * np.dot(a, b.T)

    dist = np.round(np.sqrt(p1 + p2 + p3), decimals)
    dist.flat[::dist.shape[0] + 1] = 0.0

    return dist


def compute_dist(files, type, max_dist=50, hems='L', dense=True):
    dist = []
    if files is None:
        file_name = 'parcellations/fs_LR_32k template/fs_LR.32k.%s.sphere.surf.gii' % hems
    else:
        file_name = files

    if type is 'euclidean':
        mat = nb.load(file_name)
        surf = [x.data for x in mat.darrays]
        surf_vertices = surf[0]

        dist = euclidean_distance(surf_vertices, surf_vertices)
        dist[dist > max_dist] = 0

    elif type is 'dijstra':
        mat = nb.load(file_name)
        surf = [x.data for x in mat.darrays]
        surf_vertices = surf[0]

        #TODO: call the calculation for dijstra's algorithm

    return sparse.csr_matrix(dist) if dense else dist

