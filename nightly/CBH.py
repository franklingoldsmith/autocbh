from rdkit import Chem
import igraph
import numpy as np
from copy import deepcopy
from collections import defaultdict


class buildCBH:
    def __init__(self, smile, saturate=1):
        """
        Build the CBH scheme for a given SMILES string of a molecule. \
            It gets the RDKit molecule object and graph representations of the SMILES.

        ARGUMENTS
        ---------
        :smile:     [str] SMILES string represnting a molecule
        :saturate:  [int or str] (default=1 or 'H') The integer or string representation of the \
            default molecule that will saturate the heavy atoms. \
            Usually 'H' (1), but is also often 'F' (9) or 'Cl' (17).
        """
        self.mol = Chem.MolFromSmiles(smile) # RDkit molecule object
        self.smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile)) # rewrite SMILES str in standard forms

        self.graph = mol2graph(self.mol) # Graph representation of molecule
        self.graph_adj = np.array(self.graph.get_adjacency().data) # Graph Adjacency Matrix
        self.graph_dist = np.array(self.graph.shortest_paths()) # Matrix holding distances between vertices

        # Molecule attributes with explicit hydrogens
        self.mol_h = Chem.AddHs(self.mol)
        self.graph_h = mol2graph(self.mol_h)
        self.graph_adj_h = np.array(self.graph_h.get_adjacency().data)
        self.graph_dist_h = np.array(self.graph_h.shortest_paths())

        self.cbh_pdts, self.cbh_rcts = self.build_scheme(saturate=saturate)


    def build_scheme(self, saturate=1):
        """
        Build CBH scheme and store the SMILES strings in dictionaries for each CBH level.

        ARGUMENTS
        ---------
        :saturate:  [int or str] (default=1 or 'H') The integer or string representation of the \
            default molecule that will saturate the heavy atoms. \
            Usually 'H' (1), but is also often 'F' (9) or 'Cl' (17).

        RETURNS
        -------
        (cbh_pdts, cbh_rcts)

        :cbh_pdts:  [nested dict] The product side of each CBH level \
            {cbh_level: {residual SMILES : num occurences}}
        :cbh_rcts: [nested dict] The reactant side of each CBH level \
            {cbh_level: {residual SMILES : num occurences}}
        """

        ptable = Chem.GetPeriodicTable()
        if type(saturate) == str:
                saturate = Chem.PeriodicTable.GetAtomicNumber(ptable, saturate)
        saturate_sym = Chem.PeriodicTable.GetElementSymbol(ptable, saturate)

        cbh_pdts = {} # CBH level products
        cbh_rcts = {} # CBH level reactants

        # indices of atoms where branching occurs or is a terminal 
        branch_idx = []
        branch_degrees = []
        for atom in self.mol_h.GetAtoms():
            neighbors = [1 for n in atom.GetNeighbors() if n.GetAtomicNum() != saturate]
            if len(neighbors) > 2:
                branch_idx += [atom.GetIdx()]
                branch_degrees += [len(neighbors)]

        # branch_idx = [atom.GetIdx() for atom in self.mol_h.GetAtoms() if atom.GetDegree() > 2]
        # branch_degrees = [self.mol.GetAtomWithIdx(i).GetDegree() for i in branch_idx]
        
        # Terminal atoms are those that only connected to one atom that is not the saturation atom
        # NEED to make special cases for H since it also needs to be ignored if connected to O, N etc.
        #   For now, added additional if statement: (if not hydrogenated and atom is H and if atom is connected to C)
        important_atoms = [atom for atom in self.mol_h.GetAtoms() \
            if atom.GetAtomicNum() != saturate or (saturate != 1 and atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetAtomicNum() == 6)]
        
        important_idx = [atom.GetIdx() for atom in important_atoms]
        terminal_idx = [atom.GetIdx() for atom in important_atoms if atom.GetDegree() == 1]

        cbh_level = 0
        # As long as the original molecule is not recreated in CBH levels
        while True:
            # 1. CBH products
            new_branches = [] # initialize branching
            #   a) Even cbh level --> atom centric
            if cbh_level % 2 == 0: 
                # Added important_idx instead of empty list
                residuals = self.atom_centric(cbh_level/2, True, important_idx, saturate)
                terminals = []

            #   b) Odd cbh level --> bond centric
            else: 
                residuals = self.bond_centric(np.floor(cbh_level/2), True, saturate)
                # if branch_idx has values
                if len(branch_idx) != 0: 
                    branches = self.atom_centric(np.floor(cbh_level/2), True, branch_idx, saturate)
                    for i in range(len(branches)):
                        for j in range(branch_degrees[i]-2):
                            new_branches.append(branches[i])
                    
                # Account for terminal atoms
                # if there are no terminal_idx (ie, ring), skip
                if len(terminal_idx) != 0: 
                    terminals = self.atom_centric(np.floor(cbh_level/2), True, terminal_idx, saturate)
                else:
                    terminals = []
            
            # End loop if the target molecules shows up on the product side
            if True in [Chem.MolFromSmiles(r).HasSubstructMatch(self.mol) for r in set(residuals+terminals)]:
                break
            
            # Count num of each residual and return as dictionary

            cbh_rcts[cbh_level+1] = self.count_repeats(residuals)
            cbh_pdts[cbh_level] = self.count_repeats(residuals+terminals)

            # 2. CBH reactants
            if cbh_level == 0:
                # NEED TO ADJUST THIS SECTION FOR USING DIFFERENT SATURATION ATOMS
                # NEED TO ADD STUFF SO IT ALSO CALCULATES THE NUMBER OF HF NEEDED

                # Get number of H in product (bakes in stoichiometry)
                pdt_H = sum([cbh_pdts[0][smile]*Chem.MolFromSmiles(smile).GetAtomWithIdx(0).GetTotalNumHs() \
                    for smile in cbh_pdts[0].keys()])
                # Get number of H in target molecule
                rct_H = sum([a.GetTotalNumHs() for a in self.mol.GetAtoms()])
                cbh_rcts[0] = {'[H][H]':(pdt_H - rct_H)/2}
                
                # Generatlize for any saturation atom!!!!
                if saturate != 1:
                    pdt_F = sum([cbh_pdts[0][smile]*smile.count(saturate_sym) for smile in cbh_pdts[0].keys()])
                    rct_F = sum([cbh_rcts[0][smile]*smile.count(saturate_sym) for smile in cbh_rcts[0].keys()])
                    rct_F += self.smile.count(saturate_sym)
                    cbh_rcts[0][f'{saturate_sym}{saturate_sym}'] = (pdt_F - rct_F)/2
            else:
                # Get the previous products + branch
                # cbh_rcts[cbh_level] = deepcopy(cbh_pdts[cbh_level-1])
                if len(new_branches) != 0:
                    cbh_rcts[cbh_level] = self.add_dicts(cbh_rcts[cbh_level], self.count_repeats(new_branches))

            # 3. Simplify chemical equations --> generate stoichiometry
            all_residuals = list(cbh_pdts[cbh_level].keys()) + list(cbh_rcts[cbh_level].keys())
            mols_with_repeats = [i for i in list(set(all_residuals)) if all_residuals.count(i) > 1]
            for mol_w_r in mols_with_repeats:
                num_diff = cbh_pdts[cbh_level][mol_w_r] - cbh_rcts[cbh_level][mol_w_r]
                # more of this molecule in the products
                if num_diff > 0:
                    cbh_pdts[cbh_level][mol_w_r] = num_diff
                    del cbh_rcts[cbh_level][mol_w_r]
                # more of this molecule in the reactants
                elif num_diff < 0:
                    cbh_rcts[cbh_level][mol_w_r] = -1*num_diff
                    del cbh_pdts[cbh_level][mol_w_r]
                # equal number of this molecule on both sides
                elif num_diff == 0:
                    del cbh_pdts[cbh_level][mol_w_r]
                    del cbh_rcts[cbh_level][mol_w_r]

            cbh_level += 1

        return cbh_pdts, cbh_rcts


    def atom_centric(self, dist, return_smile=False, atom_indices=[], saturate=1) -> list:
        """
        Returns a list of RDkit molecule objects that are the residuals species produced by \
            atom-centric CBH steps.

        ARGUMENTS
        ---------
        :dist:          [int] The radius or number of atoms away from a target atom (0,1,...)
        :return_smile:  [bool] Return SMILES string instead of RDKit.Chem Mol object
        :atom_indices:  [list] (default=[]) Specific atom indices to find atom-centric residuals

        RETURNS
        -------
        :residuals: [list] A list of RDkit molecule objects or SMILES strings of residual species
        """
        residuals = [] # will hold residual species

        if len(atom_indices) != 0:
            idxs = atom_indices
        else:
            idxs = range(len(self.graph_adj))
        # cycle through relevant atoms
        for i in idxs:
            if False:
                continue
            else:
                # atom indices that are within the given radius
                atom_inds = np.where(self.graph_dist_h[i] <= dist)[0]
                # create rdkit mol objects from subgraph
                residual = graph2mol(self.graph_h.subgraph(atom_inds))
                if saturate != 1:
                    # This doesn't work because it still can't differentiate implicit valence/hydrogens that
                    #   are part of the molecule or were added as a saturation atom.
                    # Would still have to AddHs to target molecule at the very beginning.
                    impl_valence = {atom.GetIdx() : atom.GetImplicitValence() for atom in residual.GetAtoms() \
                        if atom.GetImplicitValence() > 0 and atom.GetAtomicNum() in (1,6)}
                    residual = self.replace_atoms(residual, impl_valence, 1, saturate)

                if return_smile:
                    # MolToSmiles repeated to make explicit H's --> implicit
                    residual = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(residual)))
                residuals.append(residual)
        return residuals


    def bond_centric(self, dist, return_smile=False, saturate=1) -> list:
        """
        Returns a list of RDkit molecule objects that are the residuals species produced by \
            bond-centric CBH steps.

        ARGUMENTS
        ---------
        :dist:          [int] The radius or number of bonds away from a target bond (0,1,...)
        :return_smile:  [bool] Return SMILES string instead of RDKit.Chem Mol object

        RETURNS
        -------
        :residuals: [list] A list of RDkit molecule objects or SMILES strings of residual species
        """
        residuals = [] # will hold residual species
        dist += 1 # a specified distance of 0 implies the given edge --> this ensures this adjustment is made
        # cycle through edges
        for e in self.graph_h.es():
            if self.graph_h.vs()[e.source]['AtomicNum'] == saturate or self.graph_h.vs()[e.target]['AtomicNum'] == saturate:
                continue # skip edge if it's connected to a saturation atom
            # elif (self.graph_h.vs()[e.source]['AtomicNum'] == 1 or self.graph_h.vs()[e.target]['AtomicNum'] == 1) and \
            #     dist >= 2 and (self.graph_h.vs()[e.source]['AtomicNum'] != 6 or self.graph_h.vs()[e.target]['AtomicNum'] != 6):
            #     continue
            else:
                # get shortest paths from edge source and target
                gsp_s, gsp_t = self.graph_h.get_shortest_paths(e.source), self.graph_h.get_shortest_paths(e.target)
                # indices of the shortest paths that are less than 'dist' away from 'e'
                gsp_si, gsp_ti = np.where(np.array([len(x) for x in gsp_s])<=dist)[0], \
                    np.where(np.array([len(x) for x in gsp_t])<=dist)[0]
                # refine 'gsp_s/t' to only include relevant arrays
                gsp_s_refine, gsp_t_refine = (gsp_s[i] for i in gsp_si), (gsp_t[i] for i in gsp_ti)
                # define edge indices
                edge_inds = list(set([x[-1] for x in gsp_s_refine] + [x[-1] for x in gsp_t_refine]))
                # create rdkit mol objects from subgraph
                residual = graph2mol(self.graph_h.subgraph(edge_inds))
                if saturate != 1:
                    # This doesn't work because it still can't differentiate implicit valence/hydrogens that
                    #   are part of the molecule or were added as a saturation atom.
                    # Would still have to AddHs to target molecule at the very beginning.
                    impl_valence = {atom.GetIdx() : atom.GetImplicitValence() for atom in residual.GetAtoms() \
                        if atom.GetImplicitValence() > 0 and atom.GetAtomicNum() == 6}
                    residual = self.replace_atoms(residual, impl_valence, 1, saturate)
                if return_smile:
                    residual = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(residual)))
                residuals.append(residual)
        return residuals


    @staticmethod
    def count_repeats(ls) -> dict:
        """
        Count the number of repeated molecules in a given list.

        ARGUMENTS
        ---------
        :ls:    [list] list to check if there are repeated elements.

        RETURNS
        -------
        :repeats: [dict] dictionary {element : num repeats}
        """
        repeats = {i:ls.count(i) for i in list(set(ls))}
        return repeats


    @staticmethod
    def add_dicts(dict1, dict2):
        """
        Add the values within a dictionary together for matching keys.
        All dictionaries have the form: {residual SMILES : num occurences}}

        ARGUMENTS
        ---------
        :dict1: [dict] the first dictionary
        :dict2: [dict] the second dictionary 

        RETURNS
        -------
        :out_dict: [dict] the output dictionary where the number of matching SMILEs are added together
        """
        dd = defaultdict(list) # dictionary with elements of a list
        for d in (dict1, dict2):
            for key, value in d.items():
                # {key: [val1, val2]}
                dd[key].append(value)
        # add up the values for each key
        out_dict = {key:sum(dd[key]) for key in dd.keys()}
        return out_dict

    @staticmethod
    def replace_atoms(mol, dictionary, target, change):
        """
        :dictionary: [dict] {atom idx : implicit valence}
        :target:     [int]
        :change:     [int]

        :new_mol:   [RDkit Mol]
        """
        
        new_mol = Chem.RWMol(Chem.AddHs(mol))
        for i_atom, impl_val in dictionary.items():
            atom =  new_mol.GetAtomWithIdx(i_atom)
            keep_track_impl_val = 0
            for nbr in atom.GetNeighbors():
                # if neighbor is carbon or was a terminal atom
                if nbr.GetAtomicNum() == target: 
                    nbr.SetAtomicNum(change)
                    keep_track_impl_val += 1
                    # stop replacing atoms once the num of replaced exceeds the num implicit valence
                    if keep_track_impl_val >= impl_val:
                        break
        Chem.SanitizeMol(new_mol)
        return new_mol


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
    :return_smile:  [bool] (default=False) return a SMILES string instead of a RDKit.Chem Mol object.
    
    RETURNS
    -------
    :mol:   Molecular representation as either a RDkit.Chem Mol object or SMILES string
    """
    mol = Chem.rdchem.RWMol()
    # Add each vertex as an Atom
    for v in graph.vs():
        mol.AddAtom(Chem.Atom(v["AtomicNum"]))
    # Add each edge as a Bond
    for e in graph.es():
        mol.AddBond(e.source, e.target, e['BondType'])
    mol = mol.GetMol()
    Chem.SanitizeMol(mol) # ensure implicit hydrogens are accounted
    
    # Generates SMILES str
    if return_smile:
        mol = Chem.MolToSmiles(mol)
    return mol


def main():
    cbh = buildCBH('CCC(F)(F)C(F)(F)C(C(O)=O)(F)(F)', 9)
    for rung in cbh.cbh_pdts:
        print(rung, cbh.cbh_pdts[rung])

if __name__ == '__main__':
    main()
