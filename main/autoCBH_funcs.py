import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import igraph

def mol2graph(mol):
    """Molecule to Graph
    Converts a RDkit.Chem Mol object into an undirected igraph object.
    Sourced directly from: https://iwatobipen.wordpress.com/2018/05/30/active-learning-rdkit-chemoinformatics-ml/
    
    ARGUMENTS
    ---------
    :mol: RDkit.Chem Mol object - represents molecule
    
    RETURNS
    -------
    :g: igraph Graph object - chemical graph of molecule
        attributes include: atom idx, atom atomic numbers, atomic symbols, bond start/end atom idx's, bond types
    """
    # Gather atom/bond attributes
    # atom attributes: atom number, atomic number, atom symbol
    # bond attributes: atom 1 index, atom 2 index, atom bond type as number
    atom_attributes = [(a.GetIdx(), a.GetAtomicNum(), a.GetSymbol()) for a in mol.GetAtoms()]
    bond_attributes = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType(), b.GetBondTypeAsDouble()) 
                       for b in mol.GetBonds()]
    # generate chemical graph
    g = igraph.Graph()
    # Create vertices for each Atom
    for a_attr in atom_attributes:
        g.add_vertex(a_attr[0], AtomicNum=a_attr[1], AtomicSymbol=a_attr[2])
    # Create edges for each Bond
    for b_attr in bond_attributes:
        g.add_edge(b_attr[0], b_attr[1], BondType=b_attr[2], BondTypeDouble=b_attr[3])
    return g 

def graph2mol(graph, return_smile=False): 
    """Graph to Molecule
    Converts undirected igraph object to RDkit.Chem Mol object.
    Sourced directly from: https://iwatobipen.wordpress.com/2018/05/30/active-learning-rdkit-chemoinformatics-ml/
    
    ARGUMENTS
    ---------
    :graph:         [igraph graph obj] - chemical graph of molecule
        attributes include: atom idx, atom atomic numbers, atomic symbols, bond start/end atom idx's, bond types
    :return_smile:  [bool] (default=False) return a SMILE string instead of a RDKit.Chem Mol object.
    
    RETURNS
    -------
    :mol:   Molecular representation as either a RDkit.Chem Mol object or SMILES string
    """
    emol = Chem.rdchem.RWMol()
    # Add each vertex as an Atom
    for v in graph.vs():
        emol.AddAtom(Chem.Atom(v["AtomicNum"]))
    # Add each edge as a Bond
    for e in graph.es():
        emol.AddBond(e.source, e.target, e['BondType'])
    mol = emol.GetMol()
    # Generates SMILES str then converts back to Mol object 
    # (hack to ensure implicit hydrogens are accounted)
    if return_smile:
        mol = Chem.MolToSmiles(mol)
    else:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return mol

