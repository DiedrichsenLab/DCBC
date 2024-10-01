# Examples

The `examples` folder contains some showcase examples for different usage:
- `replication.py` is the script to replicate the DCBC calculation reported on the paper, which packaged to evaluate 
parcellations on `MDTB` dataset.
- `example_cortex.ipynb` is the notebook to show step-by-step DCBC calculation for a given parcellation that defined in 
**surface** representation. It also packaged to be evaluated on `MDTB` dataset.
- `example_volume.ipynb` is the notebook to show step-by-step DCBC calculation for a given parcellation that defined in
**volumetric** representation.
- `example_gpu.ipynb` is the notebook to show how to speed up DCBC calculation using GPU acceleration.
- `example.py` is the script to use compare CPU vs GPU version when evaluating a cortical parcellation (Higher GPU memory requirement)

We recommend users to check above examples in `example_cortex.ipynb` -> `example_volume.ipynb` -> `example_gpu.ipynb` order
before using the toolbox.

### 1. `example_cortex.ipynb`
This example shows step-by-step DCBC calculation for a given parcellation that defined in **surface** representation. For
example, `Power 2011` group parcellation. Then, we wanted to know how well this parcellation to predict the functional
boundaries for the MDTB subjects? To this end, this example utilizes the class `DCBC` defined in `replication.py` which
is packaged to evaluate a given parcellation on the `MDTB` data only. For simplicity, we only evaluate the left hemisphere 
as an example. Note, this example only provide a numpy CPU version computation.

### 2. `example_volume.ipynb`
Sometimes, one may want to evaluate parcellations defined in volume space or to evaluate on a custom dataset. To this end,
this example shows how user can calculate DCBC for a given parcellation in a flexible fashion. Specifically, we replaced 
the well-packaged `DCBC` class to several flat functions for custom DCBC evaluation, including

- `compute_DCBC()`: defined in `dcbc.py` which takes inputs:
    - `maxDist`: The maximum distance for vertices pairs, default 35 mm
    - `binWidth`: The spatial binning width in mm, default 1 mm
    - `parcellation`: A 1-d vector of brain parcellation to be evaluated
    - `func`: the functional data for evaluating, shape `N x P`, where `N` indicates the dimensionality of underlying 
              data, i.e. the number of task contrasts or the number of resting-state networks. `P` is the number of 
              brain voxels / vertices
    - `dist`: the pairwise distance matrix between P brain locations. It can be a dense matrix or sparse tensor
    - `weighting`: If True, the DCBC result is weighted averaged across spatial bins. If False, it is plain averaged.
    - `backend`: can be either `backend='numpy'`, so that the following calculations are using numpy cpu. Or `backend='torch'`
       to use PyTorch library for the DCBC calculation and potentially GPU acceleration.
- `compute_dist()` defined in `utilities.py` which takes inputs:
    - `dist`: the matrix of all N voxels coordinates x,y,z. Shape `N x 3`
    - `resolution`: the resolution of the volume space. Default 2*2*2 mm
    - `backend`: the backend for the calculation. If "numpy", then following 
                 calculation will be using numpy. If "torch", then following 
                 calculation will be on PyTorch.
  
This example also compatible with custom evaluation for cortical surface parcellations. Users can easilly replace `parcellation`,
`func`, and `dist` will the corresponding surface format since the calculation is the same. Remember to make sure all input
are defined using the same `backend`.

### 3. `example_gpu.ipynb`
Lastly, we provide a GPU acceleration version to dramatically speed up the DCBC calculation. In many cases, users may want to
evaluate a group parcellations on a ver large sample size, such as HCP S1200, UKB 50,000 level. The running time
for such a large scale evaluation will be extremely slow on CPU. Therefore, we made the DCBC compatible with PyTorch CUDA
tensor computation since 2.0 Release. 

For users wanted to try the GPU version, please find this notebook for a direct comparison.

### 4. PyTorch 1.0 vs. 2.0
The DCBC toolbox was written in PyTorch 1.13 + CUDA 1.16 version. Although the newer version of PyTorch is
backward compatible, there still some minor differences when defining the CUDA device global flags.

In PyTorch 1.0, the default tensor type can be defined as:

    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                               else torch.FloatTensor)

where `torch.cuda.FloatTensor` represents the tensor is allocated on CUDA device and float32 automatically. However,
in PyTorch >= 2.0, this is depreciated and replaced with more flexible way to define the device and data type separately, as

    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)

So that the tensor can be also defined on other High performance computing unit (i.e. Apple M1/M2 chip), like

    torch.set_default_device('mps')

Another new feature made in PyTorch 2.0 is that it enables the sparse tensor computation with dense tensor. Therefore,
by theory, the covariance and variance matrix calculation for the test data can be further speed-up. We welcome any 
collaboration and repository maintenance to make the DCBC calculation better.

## License

Please find out our development license (MIT) in `LICENSE` file.

## Bugs and questions

Please contact Da at dzhi@uwo.ca if you have any questions about this repository.

