#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Feb 25 10:16:23 2021

The function of calculating similarity matrix between each pair
of vertices on the cortical surface

INPUTS:     files:      The file path of geometrical information of the cortical
                        surface.
                        The default file type is 'surf.gii' as the project is
                        built upon HCP fs_LR surface space. Currently not
                        support other types of geometry files
            type:       The algorithm used to calculate the similarity between
                        vertex pairs
            hems:       The hemisphere 'L' - left; 'R' - right
            dense:      boolean variable, if True, then output the sparse matrix
                                          otherwise, output original matrix

OUTPUT:     The expected similarity (dense) matrix using given algorithm
            shape [N, N] where N indicates the size of vertices, and the
            value at N_i and N_j is the distance of the vertices i and j

Author: Da Zhi
'''
import numpy as np
import nibabel as nb
from scipy import sparse


def cosine(a, b=None):
    """
    Compute cosine similarity between samples in a and b.
        K(X, Y) = <X, Y> / (||X||*||Y||)

        :param a: ndarray, shape: (n_samples, n_features) input data.
                  e.g (32492, 34) means 32,492 cortical nodes with 34 task
                  condition activation profile
        :param b: ndarray, shape: (n_samples, n_features) input data.
                  If None, b = a

        :return:  r - the cosine similarity matrix between nodes. [N * N]
                  N is the number of cortical nodes
    """
    if b is None:
        b = a

    a_norms = np.einsum('ij,ij->i', a, a)
    b_norms = np.einsum('ij,ij->i', b, b)
    r = np.dot(a_norms, b_norms.T)

    return r


def compute_similarity(files=None, type='cosine', hems='L', dense=True, mean_centering=True):
    dist = []
    if files is None:
        file_name = 'data/group.%s.wbeta.32k.func.gii' % hems
    else:
        file_name = files

    if type is 'cosine':
        mat = nb.load(file_name)
        data = [x.data for x in mat.darrays]
        data = np.reshape(data, (len(data), len(data[0])))
        data = data.transpose()

        # mean centering of the subject beta weights
        if mean_centering:
            mean = data.mean(axis=1)
            data = data - mean[:, np.newaxis]  # mean centering
        else:
            data = data

        dist = cosine(data)

    elif type is 'pearson':
        mat = nb.load(file_name)
        data = [x.data for x in mat.darrays]
        data = np.reshape(data, (len(data), len(data[0])))
        data = data.transpose()

        # mean centering of the subject beta weights
        if mean_centering:
            mean = data.mean(axis=1)
            data = data - mean[:, np.newaxis]  # mean centering
        else:
            data = data

        dist = np.corrcoef(data)

    return sparse.csr_matrix(dist) if dense else dist


A = compute_similarity(files='data/s02/s02.L.wbeta.32k.func.gii', type='pearson', dense=False)
print(A)

