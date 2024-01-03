# autoCBH
Automated generation of Connectivity-Based Hierarchies (CBH) for selected molecules. Using these schema, heats of formation can be computed at higher accuracy. This package automates the whole process of this calculation once the neccessary experimental heats of formation and quantum mechanical energies are compiled by the user. This package is specifically novel in its ability to generate and compute heats of formation using combinations of different saturation schemes.

![General CBH schematic](figures/CBH_schematic.png#center)

## Installation
### Environment Setup
Commands to setup Python environment
#### RDKit 2022.03.5
```
conda create -n autocbh python=3.9 numpy=1.23 pandas=1.4 rdkit=2022.03.5 python-igraph=0.9.11 pygraphviz=1.9 networkx notebook matplotlib pytest tqdm pyyaml -c defaults -c conda-forge -c anaconda
```

#### RDKit 2022.09 (changes to $\texttt{CanonSmiles}$ algorithm)
```
conda create -n testautocbh python=3.9 numpy=1.23 pandas=1.4 rdkit=2022.09 python-igraph=0.9.11 pygraphviz=1.9 networkx notebook matplotlib pytest tqdm pyyaml -c defaults -c conda-forge -c anaconda
```
### Installation of package
TBA

## Features
#### 1. Automated CBH scheme generation
![GenX CBH-2](figures/genx_cbh2.png)

#### 2. Automated calculation of heats of formation across multiple levels of theory
![Example output dataframe](figures/output_dataframe_example.png)

#### 3. Error logging during calculation
![Error logging](figures/print_errors.png)

#### 4. Thermochemical network visualization
![PFOA TN](figures/TN_PFOA_H.png)

#### 5. Uncertainty propagation quantification
![UQ](figures/UQ_genx_rel.png)