# autoCBH
Automated generation of Connectivity-Based Hierarchies (CBH) for selected molecules.

## Environment Setup
Commands to setup Python environment
### RDKit 2022.03.5
```
conda create -n autocbh python=3.9 numpy=1.23 pandas=1.4 rdkit=2022.03.5 python-igraph=0.9.11 pygraphviz=1.9 networkx notebook matplotlib pytest tqdm pyyaml -c defaults -c conda-forge -c anaconda
```

### RDKit 2022.09 (changes to $\texttt{CanonSmiles}$ algorithm)
```
conda create -n testautocbh python=3.9 numpy=1.23 pandas=1.4 rdkit=2022.09 python-igraph=0.9.11 pygraphviz=1.9 networkx notebook matplotlib pytest tqdm pyyaml -c defaults -c conda-forge -c anaconda
```