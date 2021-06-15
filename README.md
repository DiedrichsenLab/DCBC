## DCBC evaluation

Diedrichsen Lab, Western University

This repository is the toolbox of the paper "Evaluating brain parcellations using the distance controlled boundary coefficient". It contains all the functions needed to evaluation given cortical parcellations. See the [paper](https://www.biorxiv.org/content/10.1101/2021.05.11.443151v1) for more details.

### Installation and dependencies
This project depends on several third party libraries, including: [nibabel](https://nipy.org/nibabel/) (version>=2.4), [scipy](https://www.scipy.org/) (version>=1.3.1), [numpy](https://numpy.org/) (version>=1.17.4), and [matplotlib](https://matplotlib.org/) (version>=3.0.2)

	pip install nibabel scipy numpy matplotlib


Or you can install the package manually from the original binary source as above links.	

### Structure of the DCBC project

The default structure of DCBC project, including the scripts, parcellations, raw fMRI data, and etc. as follows

    project/
    │   README.md
    │   eval_DCBC.py
    │   plotting.py
    │   ...
    │   sample.ipynb
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

**1. Function files description**

The project is designed in a flatten fashion where all needed functions are stored in each .py files under root directory, including,

`compute_distance.py` contains the necessary functions to compute distance metrics of expected kernel.\
`compute_similarity.py` is used to calculate similarity matrix between each of the brain location pairs.\
`eval_DCBC.py` is the main entry of the DCBC evaluation project which contains the DCBC class and its function to evaluate a given parcellation.\
`plotting.py` is the helper function that provides plotting method to decode evaluation result from previous step, and plot the within- and between-parcel correlations with respect to bins.

**2. Subject data structures**

There are 24 subjects data from [MDTB](https://openneuro.org/datasets/ds002105/versions/1.1.0) dataset stored in folder `data`. In each of the subjects sub-folder, it contains two files of left and right-hemisphere of this subject in standard HCP fs-LR 32k template space.

If users want to use own subjects data, please copy to `data` folder and remain the current structures. Because the DCBC evaluation function automatically scan all sub-folders in it and read their hemisphere data.

Each of the hemisphere data has a shape of `(N, k)` matrix, where `N` indicates the number of vertices or brain locations, `k` represents the dimensions of features. In our case, `k = 34` which mean there are 34 task conditions pre-whitenned beta values. And it can be changed to any connectivity measures, such as time-series.

**3. Distance metrics**

User can put own distance metrics file into `distanceMarix` folder. Unfortunately, github is not allow us to upload large files but you can still use `compute_distance` function to calculate your own distance metrics.

For download our pre-computed distance that used in the paper, please go to [distances](http://www.diedrichsenlab.org/).


**4. Parcellations**

We summarized several commonly-used cortical parcellations and converted them all into standard HCP fs-LR 32k template in `parcellations` folder. 

You can also find the 32k template files `fs_LR.32k.X.sphere.surf.gii` or `fs_LR.32k.X.midthickness.surf.gii` in the sub-folder. We also organised some parcellations if they provide multiple resolutions and it's a good example to test DCBC evaluation is robust across different resolutions.


### Usage example

Please run our sample code `sample.py` or go through the notebook with inline results in `sample.ipynb` for more details.

### Reference and License

Please find out our development license (MIT) in `LICENSE` file.

### Bugs and questions

Please contact Da at dzhi@uwo.ca if you have any questions about this repository.

