#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The collection of helper functions for the DCBC calculation

Created on Fri Sep 27 11:55:06 2024
Author: dzhi
"""
import os, warnings, scipy
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# Check if torch is available
try:
    import torch as pt
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


def sub2ind(array_shape, rows, cols):
    """ Convert subscripts to linear indices

    Args:
        array_shape: shape of the array
        rows: row subscripts
        cols: colum subscripts

    Returns:
        linear indices
    """
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    """ Convert linear indices to subscripts

    Args:
        array_shape: shape of the array
        ind: linear indices

    Returns:
        rows: row subscripts
        cols: column subscripts
    """
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


def scan_subdirs(path):
    """ Scan directory and get all visible folder names.

    Args:
        path: The directory path to be scanned

    Returns:
        a list of folder names
    """
    sub_dirs = []
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            sub_dirs.append(entry.name)

    return sub_dirs


def euclidean_distance(a, b=None, decimals=3, max_dist=None):
    """
    Compute euclidean similarity between samples in a and b 
    incrementally to save memory.

    Args:
        a (ndarray): shape of (n_samples, n_features) input data. 
                     e.g (32492, 34) means 32,492 cortical nodes with 
                     34 tasks 
            condition activation profile
        b (ndarray): shape of (n_samples, n_features) input data. 
                     If None, b = a.
        decimals: the precision when rounding.
        max_dist: Optional; distances greater than this value will be 
                  set to 0.

    Returns:
        r: the pairwise distance matrix [N x N], where N is the number 
           of samples in a.
    """
    if b is None:
        b = a

    N = a.shape[0]
    dist = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        # Compute distances incrementally for row 'i'
        diff = a[i] - b
        row_dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        dist[i] = np.round(row_dist, decimals)

        # Apply max_dist condition if provided
        if max_dist is not None:
            dist[i][dist[i] > max_dist] = 0

    return dist


def compute_dist_from_surface(files, type, max_dist=50, hems='L', sparse=True):
    """ The function of calculating distance matrix between each pair
        of vertices on the cortical surface

    Args:
        files: The geometrical information of the cortical surface.
               The default file type is 'surf.gii' as the project is
               built upon HCP fs_LR surface space. It's currently not
               support other types of geometry files
        type: The algorithm used to calculate the distance between
              vertex pairs
        max_dist: the maximum distance of the pairwise vertices. Set to
                  0 if the distance of a vertex pair > max_dist
        hems: specify the hemisphere to compute
        sparse: return a sparsed matrix if True, Else, dense matrix

    Returns:
        The expected distance matrix using given algorithm
        shape [N, N] where N indicates the size of vertices, and the
        value at N_i and N_j is the distance of the vertices i and j
    """
    dist = []
    if files is None:
        file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'parcellations', 'fs_LR_32k_template',
                                 f'fs_LR.32k.{hems}.sphere.surf.gii')
    else:
        file_name = files

    if type == 'euclidean':
        mat = nb.load(file_name)
        surf = [x.data for x in mat.darrays]
        surf_vertices = surf[0]

        dist = euclidean_distance(surf_vertices, surf_vertices,
                                  max_dist=max_dist)
    elif type == 'dijstra':
        mat = nb.load(file_name)
        surf = [x.data for x in mat.darrays]
        surf_vertices = surf[0]
        # TODO: call the calculation for dijstra's algorithm

    return scipy.sparse.csr_matrix(dist) if sparse else dist


def convert_numpy_to_torch_sparse(dist, device='cpu'):
    """Convert the numpy sparse matrix to a PyTorch coo sparse tensor

    Args:
        dist (numpy.ndarray): input distance matrix
        device: transfer the resulting sparse matrix to a given device

    Returns:
        a sparsed tensor defined on a given device
    """
    # assert scipy.sparse.issparse(dist), "The input should be a sparse matrix"

    coo_matrix = dist.tocoo()
    indices = pt.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = pt.FloatTensor(coo_matrix.data)
    shape = coo_matrix.shape
    sparse_tensor = pt.sparse_coo_tensor(indices, values, pt.Size(shape))

    return sparse_tensor.to(device=device)


def cosine(a, b=None):
    """ Compute cosine similarity between samples in a and b.
        K(X, Y) = <X, Y> / (||X||*||Y||)

    Args:
        a (ndarray): shape of (n_samples, n_features) input data.
                     e.g (32492, 34) means 32,492 cortical nodes
                     with 34 task condition activation profile
        b (ndarray): shape of (n_samples, n_features) input data.
                     If None, b = a

    Returns:
        r: the cosine similarity matrix between nodes. [N * N]
           N is the number of cortical nodes
    """
    if b is None:
        b = a

    a_norms = np.einsum('ij,ij->i', a, a)
    b_norms = np.einsum('ij,ij->i', b, b)
    r = np.dot(a_norms, b_norms.T)

    return r


def compute_similarity(files=None, type='cosine', hems='L', sparse=True, mean_centering=True):
    """ The function of calculating similarity matrix between each pair
        of vertices on the cortical surface (Replication only)

    Args:
        files: the file path of geometrical information of the cortical surface.
               The default file type is 'surf.gii' as the project is
               built upon HCP fs_LR surface space. It's currently not
               support other types of geometry files
        type: the algorithm used to calculate the similarity between
              vertex pairs
        hems: The hemisphere 'L' - left; 'R' - right
        sparse: boolean variable, if True, then output the sparse matrix
                Otherwise, output original matrix
        mean_centering: If True, demean the input functional data

    Returns:
        The expected similarity matrix (sparse or dense) using given algorithm
        shape [N, N] where N indicates the size of vertices, and the
        value at N_i and N_j is the distance of the vertices i and j
    """
    dist = []
    if files is None:
        file_name = '../data/group.%s.wbeta.32k.func.gii' % hems
    else:
        file_name = files

    if type == 'cosine':
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

    elif type == 'pearson':
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

    return scipy.sparse.csr_matrix(dist) if sparse else dist


def compute_dist(coord, resolution=2, backend='torch'):
    """ Calculate the distance matrix between each of the voxel pairs by given mask file
        Automatically chooses the backend or uses user-specified backend.

    Args:
        coord: the matrix of all N voxels coordinates x,y,z. Shape N * 3
        resolution: the resolution of .nii file. Default 2*2*2 mm
        backend: the backend for the calculation. If "numpy", then following
                 calculation will be using numpy. If "torch", then following
                 calculation will be on PyTorch.

    Returns:
        a distance matrix of N * N, where N represents the number of masked voxels
    """
    if backend == 'torch' and TORCH_AVAILABLE:
        if type(coord) is np.ndarray:
            coord = pt.tensor(coord, dtype=pt.get_default_dtype())
        assert type(coord) is pt.Tensor, "Input coord must be pytorch tensor!"
        return compute_dist_pt(coord, resolution=resolution)
    elif backend == 'numpy' or not TORCH_AVAILABLE:
        return compute_dist_np(coord, resolution=resolution)
    else:
        raise ValueError("Torch not available and no valid backend specified!")


def compute_dist_pt(coord, resolution=2):
    """ Calculate the distance matrix between each of the voxel pairs by given mask file
        (PyTorch version)

    Args:
        coord: the tensor of all N voxels coordinates x,y,z. Shape N * 3
        resolution: the resolution of .nii file. Default 2*2*2 mm

    Returns:
        a distance tensor of N * N, where N represents the number of masked voxels
    """
    if type(coord) is np.ndarray:
        coord = pt.tensor(coord, dtype=pt.get_default_dtype())

    num_points = coord.shape[0]
    D = pt.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return pt.sqrt(D) * resolution


def compute_dist_np(coord, resolution=2):
    """ Calculate the distance matrix between each of the voxel pairs by given mask file
        (Numpy version)

    Args:
        coord: the ndarray of all N voxels coordinates x,y,z. Shape N * 3
        resolution: the resolution of .nii file. Default 2*2*2 mm

    Returns:
        a distance matrix of N * N, where N represents the number of masked voxels
    """
    num_points = coord.shape[0]
    D = np.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return np.sqrt(D) * resolution


### variance / covariance
def compute_var_cov(data, cond='all', mean_centering=True, backend='torch'):
    """ Compute the variance and covariance for a given data matrix.
        Automatically chooses the backend or uses user-specified backend.

    Args:
        data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        cond: specify the subset of activation conditions to evaluation
              (e.g condition column [1,2,3,4]), if not given, default to
              use all conditions
        mean_centering: boolean value to determine whether the given subject
                        data should be mean centered
        backend: the backend for the calculation. If "numpy", then following
                 calculation will be using numpy. If "torch", then following
                 calculation will be on PyTorch.

    Returns: cov - the covariance matrix of current subject data, shape [N * N]
             var - the variance matrix of current subject data, shape [N * N]
    """
    if backend == 'torch' and TORCH_AVAILABLE:
        if type(data) is np.ndarray:
            data = pt.tensor(data, dtype=pt.get_default_dtype())
        assert type(data) is pt.Tensor, "Input data must be pytorch tensor!"
        return compute_var_cov_pt(data, cond=cond, mean_centering=mean_centering)
    elif backend == 'numpy' or not TORCH_AVAILABLE:
        return compute_var_cov_np(data, cond=cond, mean_centering=mean_centering)
    else:
        raise ValueError("Torch not available and no valid backend specified!")


def compute_var_cov_np(data, cond='all', mean_centering=True):
    """ Compute the variance and covariance for a given data matrix.
        (Numpy CPU version)

    Args:
        data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        cond: specify the subset of activation conditions to evaluation
              (e.g condition column [1,2,3,4]), if not given, default to
              use all conditions
        mean_centering: boolean value to determine whether the given subject
                        data should be mean centered

    Returns: cov - the covariance matrix of current subject data, shape [N * N]
             var - the variance matrix of current subject data, shape [N * N]
    """
    if mean_centering:
        mean = data.mean(axis=1)
        data = data - mean[:, np.newaxis]  # mean centering
    else:
        data = data

    # specify the condition index used to compute correlation,
    # otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    k = data.shape[1]
    sd = np.sqrt(np.sum(np.square(data), axis=1) / (k-1))  # standard deviation
    sd = np.reshape(sd, (sd.shape[0], 1))
    var = np.matmul(sd, sd.transpose())
    cov = np.matmul(data, data.transpose()) / (k-1)
    return cov, var


def compute_var_cov_pt(data, cond='all', mean_centering=True):
    """ Compute the variance and covariance for a given data matrix.
        (PyTorch GPU version)

    Args:
        data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        cond: specify the subset of activation conditions to evaluation
              (e.g condition column [1,2,3,4]), if not given, default to
              use all conditions
        mean_centering: boolean value to determine whether the given subject
                        data should be mean centered

    Returns: cov - the covariance matrix of current subject data, shape [N * N]
             var - the variance matrix of current subject data, shape [N * N]
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


def plot_single(within, between, subjects, maxDist=35, binWidth=1,
                within_color='k', between_color='r'):
    """ helper function to plot a single DCBC results (Replication only)

    Args:
        within: within-parcel correlation values
        between: between-parcel correlation values
        subjects: list of subjects
        maxDist: the maximum distance for evaluating / plotting
        binWidth: bin width of the dcbc calculation
        within_color: with-curve color
        between_color: between-curve color

    Returns:
        plot the DCBC evaluation given the results
    """
    fig = plt.figure()

    # Obtain basic info from evaluation result
    numBins = int(np.floor(maxDist / binWidth))
    num_sub = len(subjects)
    x = np.arange(0, maxDist, binWidth) + binWidth / 2

    y_within = within.reshape(num_sub, -1)
    y_between = between.reshape(num_sub, -1)

    plt.errorbar(x, y_within.mean(0), yerr=y_within.std(0),
                 ecolor=within_color, color=within_color,
                 elinewidth=0.5, capsize=2, linestyle='dotted',
                 label='within')
    plt.errorbar(x, y_between.mean(0), yerr=y_between.std(0),
                 ecolor=between_color, color=between_color,
                 elinewidth=0.5, capsize=2, linestyle='dotted',
                 label='between')

    plt.legend(loc='upper right')
    plt.show()


def plot_wb_curve(T, path, sub_list=None, hems='all', within_color='k',
                  between_color='r'):
    """ helper function to plot within/between curves DCBC results
        (Replication only)

    Args:
        T: the DCBC evaluation results
        path: the folder path of the test data
        sub_list: a list of subjects
        hems: the hemisphere of the evaluation,
              'L' - left; or 'R' right
        within_color: with-curve color
        between_color: between-curve color

    Returns:
        plot the DCBC evaluation given the results
    """
    fig = plt.figure()

    # Obtain basic info from evaluation result T
    bin_width = [value for key, value in T.items()][0]['binWidth']
    max_dist = [value for key, value in T.items()][0]['maxDist']
    k = len([value for key, value in T.items()][0]['corr_within'])
    x = np.arange(0,max_dist,bin_width) + bin_width/2

    # if hems is 'all' and any([x for x in T.keys() if 'L' in x]) and any([x for x in T.keys() if 'R' in x]):
    #     # subjectsDir = [x for x in T.keys()]
    #     pass
    # elif hems is 'L' or 'R' and any([x for x in T.keys() if hems in x]):
    #     # subjectsDir = [x for x in T.keys() if hems in x]
    #     pass
    # else:
    #     raise TypeError("Input hemisphere's data has not been found!")

    if sub_list is not None:
        subjects_dir = sub_list
    else:
        subjects_dir = scan_subdirs(path)

    y_within, y_between = np.empty([1, k]), np.empty([1, k])
    for sub in subjects_dir:
        data = [value for key, value in T.items() if sub in key]
        if len(data) == 2 and (hems == 'all'):
            within = (np.asarray(data[0]["corr_within"]) + np.asarray(data[1]["corr_within"])) / 2
            between = (np.asarray(data[0]["corr_between"]) + np.asarray(data[1]["corr_between"])) / 2
        elif len(data) == 1 and data[0]["hemisphere"] is hems:
            within = data[0]["corr_within"]
            between = data[0]["corr_between"]
        else:
            raise Exception("Incomplete DCBC evaluation. Missing result of %s." % sub)

        y_within = np.vstack((y_within, within))
        y_between = np.vstack((y_between, between))
        sub_list = T.keys()

    y_within = np.delete(y_within, 0, axis=0)
    y_between = np.delete(y_between, 0, axis=0)

    plt.errorbar(x, y_within.mean(0), yerr=y_within.std(0), ecolor=within_color, color=within_color, label='within')
    plt.errorbar(x, y_between.mean(0), yerr=y_between.std(0), ecolor=between_color, color=between_color, label='between')
    plt.xlabel('spatial distance')
    plt.ylabel('Correlation between vertex pairs')
    plt.legend(loc='upper right')
    plt.show()
