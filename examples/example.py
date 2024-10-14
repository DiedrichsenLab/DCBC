import numpy as np
import scipy as sp
import scipy.io as spio
import mat73, time
import nibabel as nb
from utilities import compute_dist, convert_numpy_to_torch_sparse
from dcbc import compute_DCBC

try:
    import torch as pt
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : False
DEVICE = 'cuda' if pt.cuda.is_available() else 'cpu'
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)


def calc_test_dcbc(parcels, testdata, dist, max_dist=35, bin_width=1,
                   trim_nan=False, return_wb_corr=False, verbose=True):
    """DCBC: evaluate the resultant parcellation using DCBC
    Args:
        parcels (np.ndarray): the input parcellation:
            either group parcellation (1-dimensional: P)
            individual parcellation (num_subj x P )
        dist (pt.Tensor): the distance metric
        testdata (np.ndarray): the functional test dataset,
                                shape (num_sub, N, P)
        trim_nan (boolean): if true, make the nan voxel label will be
                            removed from DCBC calculation. Otherwise,
                            we treat nan voxels are in the same parcel
                            which is label 0 by default.
    Returns:
        dcbc_values (np.ndarray): the DCBC values of subjects
    """
    if trim_nan:
        idx = pt.where(~pt.isnan(parcels))[0] \
            if parcels.ndim == 1 else \
            pt.where(~pt.isnan(parcels[0]))[0]

        parcels = parcels[idx] if parcels.ndim == 1 else \
            parcels[:, idx]
        testdata = testdata[:, :, idx]
        dist = pt.index_select(dist, 0, idx)
        dist = pt.index_select(dist, 1, idx)

    dcbc_values, D_all = [], []
    for sub in range(testdata.shape[0]):
        print(f'Subject {sub}', end=':')
        tic = time.perf_counter()
        if parcels.ndim == 1:
            D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                             parcellation=parcels,
                             dist=dist, func=testdata[sub].T)
        else:
            D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                             parcellation=parcels[sub],
                             dist=dist, func=testdata[sub].T)
        dcbc_values.append(D['DCBC'])
        # within.append(pt.stack(D['corr_within']))
        # between.append(pt.stack(D['corr_between']))
        D_all.append(D)
        toc = time.perf_counter()
        print(f"{toc - tic:0.4f}s")

    if return_wb_corr:
        return pt.stack(dcbc_values), D_all
    else:
        return pt.stack(dcbc_values)


if __name__ == "__main__":
    # Load cortical parcellation from label.gii file
    gii_file = nb.load('../parcellations/Power2011.32k.L.label.gii')
    # Make sure the input parcels is a N-long vector
    parcels = gii_file.darrays[0].data

    # Load sparse distance matrix of left hemisphere
    dist = spio.loadmat('../distanceMatrix/distAvrg_sp.mat')['avrgDs']

    # Load test dataset - i.e subject 02 left hemisphere
    t_file = nb.load('../data/s02/s02.L.wbeta.32k.func.gii')
    t_data = np.stack([x.data for x in t_file.darrays])

    # Load medial wall mask / remove from computation
    mask_file_L = nb.load('../parcellations/fs_LR_32k template/fs_LR.32k.L.mask.label.gii')
    mask_L = mask_file_L.darrays[0].data.astype(bool)

    # Remove medial wall from computation
    parcels = parcels[mask_L]
    dist = dist[mask_L][:, mask_L]
    t_data = t_data[:, mask_L]

    # DCBC evaluation: evaluate Power 2011 group parcellation on MDTB subject 02
    tic = time.perf_counter()
    D = compute_DCBC(maxDist=35, binWidth=1, parcellation=parcels,
                     dist=dist, func=t_data.T, backend='numpy')
    toc = time.perf_counter()
    print(f"Numpy CPU version DCBC calculation used {toc - tic:0.4f}s")

    # Move to cuda tensor
    parcels = pt.tensor(parcels, dtype=pt.get_default_dtype())
    dist = convert_numpy_to_torch_sparse(dist, device=DEVICE)
    t_data = pt.tensor(t_data, dtype=pt.get_default_dtype())
    tic = time.perf_counter()
    D = compute_DCBC(maxDist=35, binWidth=1, parcellation=parcels,
                     dist=dist, func=t_data.T, backend='torch')
    toc = time.perf_counter()
    print(f"PyTorch GPU version DCBC calculation used {toc - tic:0.4f}s")

    print(f'Power 2011 - DCBC on subject 02 is: %f' % D['DCBC'])