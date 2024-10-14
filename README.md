# DCBC evaluation

Diedrichsen Lab, Western University

This repository is the toolbox of the paper "Evaluating brain parcellations using the distance controlled boundary coefficient". It contains all the functions needed to evaluation given cortical parcellations. See the [paper](https://www.biorxiv.org/content/10.1101/2021.05.11.443151v1) for more details.

## Reference

Please cite this article if using this toolbox:

- Zhi, D., King, M., Hernandez‐Castillo, C. R., & Diedrichsen, J. (2022). Evaluating brain parcellations using the distance‐controlled boundary coefficient. Human Brain Mapping, 43(12), 3706-3720.

## Installation and dependencies
This project depends on several third party libraries, including: [nibabel](https://nipy.org/nibabel/) (version>=2.4), [scipy](https://www.scipy.org/) (version>=1.3.1), [numpy](https://numpy.org/) (version>=1.17.4), and [matplotlib](https://matplotlib.org/) (version>=3.0.2)

	pip install nibabel scipy numpy matplotlib

Or you can install the package manually from the original binary source as above links.	
Once you have cloned this repository, you need to add it to your PYTHONPATH, so you can import the functionality.

    PYTHONPATH=<repository_path>/DCBC:${PYTHONPATH}
    export PYTHONPATH

To use GPU acceleration, please ensure to install the compatible PyTorch and CUDA version. Details can be found
at the official PyTorch webpage at https://pytorch.org/. This example was written with `torch_version=1.13` and
`CUDA_version=1.16`. For example,

    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

## Structure of the DCBC project

The default structure of DCBC project, including the scripts, parcellations, raw fMRI data, and etc. as follows

    project/
    │   README.md
    │   dcbc.py
    │   utilities.py
    │   ...
    │   __init__.py
    └───examples/
    │       example.py
    │       example_cortex.ipynb
    │       ...
    │
    └───distanceMatrix/
    │       disAvrg_sp.mat
    │       ...
    │   
    └───parcellations/
    │       └───random/
    │       │       Icosahedron-42.32k.L.label.gii
    │       │       ...
    │       AAL.32k.L.label.gii
    │       ...
    │   
    └───data/
    │       s02
    │       s03
    │       ...
    │       s31

### 1. Function files description

The project is designed in a simple flat fashion where all needed functions are stored in either

- `utilities.py` files: which contains all necessary functions to support DCBC calculation
- `dcbc.py`: contains the main (entry) function for DCBC calculation

### 2. Subject data structures

There are 24 subjects data from [MDTB](https://openneuro.org/datasets/ds002105/versions/1.1.0) dataset stored in folder `data`. In each of the subjects sub-folder, it contains two files of left and right-hemisphere of this subject in standard HCP fs-LR 32k template space.

Each of the hemisphere data has a shape of `(N, k)` matrix, where `N` indicates the number of vertices or brain locations, `k` represents the dimensions of features. In our case, `k = 34` which mean there are 34 task conditions pre-whitenned beta values. And it can be changed to any connectivity measures, such as time-series.

If users want to use own subjects data, please use the plain DCBC calculation directly. See `examples/example_volume.ipynb` or `examples/example_gpu.ipynb`

### 3. Distance metrics

User can put own distance metrics file into `distanceMarix` folder. Unfortunately, GitHub doesn't allow us to upload large files but you can still use 
- `utilities.compute_dist()` function to calculate your own volumetric distance, or
- `utilities.compute_dist_from_surface()` to calculate the surface vertices distances by a given `surf.gii`

We also provide several pre-computed distance matrix that can be used in case users don't want to wait the distance computation, 
please find more details and download at http://www.diedrichsenlab.org/toolboxes/toolbox_dcbc.htm.

### 4. Parcellations

We summarized several commonly-used cortical parcellations and converted them all into standard HCP fs-LR 32k template 
in `parcellations` folder. 

User can also find the 32k template files `fs_LR.32k.X.sphere.surf.gii` or `fs_LR.32k.X.midthickness.surf.gii` in the 
sub-folder (X = L or R, representing left or right hemisphere). We also collected some commonly-used group parcellations
in `parcellations` folder and they're good examples to test DCBC evaluation is robust across different resolutions.

We always welcome contributors to this group-level cortical parcellation collection. If you want to contribute your 
parcellation to the evaluation, please contact Diedrichsen Lab for instructions.

For users interested in cerebellar parcellations in volume space, please find more details in the 
[cerebellar atlases](https://github.com/DiedrichsenLab/cerebellar_atlases) repository.

## Usage example

Please check our sample code and notebooks in `/examples` folder. See the readme file in the folder for more details.

## License

Please find out our development license (MIT) in `LICENSE` file.

## Bugs and questions

Please contact Da Zhi [dzhi@uwo.ca]() if you have any questions about this repository.

