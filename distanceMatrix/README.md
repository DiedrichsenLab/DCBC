## Distance file

Due to the GitHub upload limit, there is no distance file in this folder. To support the implementation of 
the examples, user may want to download the pre-computed distance file to this folder from https://www.diedrichsenlab.org/toolboxes/toolbox_dcbc.htm

We provide three distance metrics for left hemisphere in fsLR_32k space, including: 

### 1.`distSphere_sp.mat`
The distance matrix between each of the vertex pairs on the cortical surface in fs-LR 32k template 
projected on the standard sphere.

### 2. `distAvrg_sp.mat`
This distance matrix is calculated using Dijkstra's algorithm (Dijkstra et al., 1959) to estimate the 
shortest paths between each pair of vertices on each individual cortical surface. We then used the 
mid-cortical layer which is the average of the pial and white-gray matter surface.

### 3. `distGOD_sp.mat`
This distance matrix is computed using Connectome Workbench command -surface-geodesic-distance to 
compute the GOD distance between each pair of the cortical vertices.