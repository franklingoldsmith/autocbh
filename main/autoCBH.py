import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import igraph
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
from autoCBH_funcs import mol2graph, graph2mol
from copy import deepcopy
from collections import defaultdict

class buildCBH:
    def __init__(self, smile):
        """
        Build the CBH scheme for a given SMILE string of a molecule. \
            It gets the RDKit molecule object and graph representations of the SMILE.

        ARGUMENTS
        ---------
        :smile:     [str] SMILE string represnting a molecule
        """
        self.mol = Chem.MolFromSmiles(smile) # RDkit molecule object
        self.smile = Chem.MolToSmiles(self.mol) # rewrite SMILE str in standard forms

        self.graph = mol2graph(self.mol) # Graph representation of molecule
        self.graph_adj = np.array(self.graph.get_adjacency().data) # Graph Adjacency Matrix
        self.graph_dist = np.array(self.graph.shortest_paths()) # Matrix holding distances between vertices

        self.cbh_pdts, self.cbh_rcts = self.build_scheme()


    def atom_centric(self, dist, return_smile=False, atom_indices=[]) -> list:
        """
        Returns a list of RDkit molecule objects that are the residuals species produced by \
            atom-centric CBH steps.

        ARGUMENTS
        ---------
        :dist:          [int] The radius or number of atoms away from a target atom (0,1,...)
        :return_smile:  [bool] Return SMILE string instead of RDKit.Chem Mol object
        :atom_indices:  [list] (default=[]) Specific atom indices to find atom-centric residuals

        RETURNS
        -------
        :residuals: [list] A list of RDkit molecule objects or SMILE strings of residual species
        """
        residuals = [] # will hold residual species

        if len(atom_indices) != 0:
            idxs = atom_indices
        else:
            idxs = range(len(self.graph_adj))
        # cycle through relevant atoms
        for i in idxs:
            # atom indices that are within the given radius
            atom_inds = np.where(self.graph_dist[i] <= dist)[0]
            # create rdkit mol objects from subgraph
            residual = graph2mol(self.graph.subgraph(atom_inds), return_smile)
            residuals.append(residual)
        return residuals


    def bond_centric(self, dist, return_smile=False) -> list:
        """
        Returns a list of RDkit molecule objects that are the residuals species produced by \
            bond-centric CBH steps.

        ARGUMENTS
        ---------
        :dist:          [int] The radius or number of bonds away from a target bond (0,1,...)
        :return_smile:  [bool] Return SMILE string instead of RDKit.Chem Mol object

        RETURNS
        -------
        :residuals: [list] A list of RDkit molecule objects or SMILE strings of residual species
        """
        residuals = [] # will hold residual species
        dist += 1 # a specified distance of 0 implies the given edge --> this ensures this adjustment is made
        # cycle through edges
        for e in self.graph.es():
            # get shortest paths from edge source and target
            gsp_s, gsp_t = self.graph.get_shortest_paths(e.source), self.graph.get_shortest_paths(e.target)
            # indices of the shortest paths that are less than 'dist' away from 'e'
            gsp_si, gsp_ti = np.where(np.array([len(x) for x in gsp_s])<=dist)[0], \
                np.where(np.array([len(x) for x in gsp_t])<=dist)[0]
            # refine 'gsp_s/t' to only include relevant arrays
            gsp_s_refine, gsp_t_refine = (gsp_s[i] for i in gsp_si), (gsp_t[i] for i in gsp_ti)
            # define edge indices
            edge_inds = list(set([x[-1] for x in gsp_s_refine] + [x[-1] for x in gsp_t_refine]))
            # create rdkit mol objects from subgraph
            residual = graph2mol(self.graph.subgraph(edge_inds), return_smile)
            residuals.append(residual)
        return residuals


    def build_scheme(self):
        """
        Build CBH scheme and store the SMILE strings in dictionaries for each CBH level.

        ARGUMENTS
        ---------
        None

        RETURNS
        -------
        :cbh_pdts:  [nested dict] The product side of each CBH level \
            {cbh_level: {residual SMILE : num occurences}}
        :branch_rcts: [nested dict] The residuals resulting from branching of the CBH scheme \
            {cbh_level: {residual SMILE : num occurences}}
        """

        cbh_pdts = {} # CBH level products
        cbh_rcts = {} # CBH level reactants
        branch_rcts = {}

        # indices of atoms where branching occurs or is a terminal 
        branch_idx = np.where(np.array([atom.GetDegree() for atom in self.mol.GetAtoms()]) > 2)[0].tolist()
        branch_degrees = [self.mol.GetAtomWithIdx(i).GetDegree() for i in branch_idx]
        terminal_idx = np.where(np.array([atom.GetDegree() for atom in self.mol.GetAtoms()]) == 1)[0].tolist()

        molec_list = ()
        cbh_level = 0
        # As long as the original molecule is not recreated in CBH levels
        while self.smile not in molec_list:
            # even cbh level --> atom centric
            if cbh_level % 2 == 0: 
                residuals = self.atom_centric(cbh_level/2, True)

            # odd cbh level --> bond centric
            else: 
                residuals = self.bond_centric(np.floor(cbh_level/2), True)
                # if branch_idx has values
                if len(branch_idx) != 0: 
                    # THIS NEEDS TO SCALE WITH atom.GetDegrees()-2
                    branches = self.atom_centric(np.floor(cbh_level/2), True, branch_idx)
                    new_branches = []
                    for i in range(len(branches)):
                        for j in range(branch_degrees[i]-2):
                            new_branches.append(branches[i])
                    branch_rcts[cbh_level] = self.count_repeats(new_branches)
                # account for terminal atoms
                if cbh_level == 1: 
                    terminals = self.atom_centric(0, True, terminal_idx)
                    residuals = residuals + terminals
            
            cbh_pdts[cbh_level] = self.count_repeats(residuals)
            
            molec_list = set(residuals)
            cbh_level += 1

        # Delete the last CBH level that contains the original molecule
        last_level = max(cbh_pdts.keys()) # CBH level that contains the original molecule
        del cbh_pdts[last_level] # delete that level
        if last_level in branch_rcts.keys(): # if there were no branches in the last CBH level
            del branch_rcts[last_level]
        
        # MIGHT BE ABLE TO PUT THIS IN THE ABOVE 'WHILE' LOOP
        # THIS WOULD ALLOW ME TO GET RID OF THE branch_rcts DICT
        # Create reactant side of CBH Schemes
        for cbh_level in range(len(cbh_pdts.keys())):
            if cbh_level == 0:
                # get number of H in product
                pdt_H = sum([cbh_pdts[0][key]*Chem.MolFromSmiles(key).GetAtomWithIdx(0).GetTotalNumHs() \
                    for key in cbh_pdts[0].keys()])
                # get number of H in target molecule
                rct_H = sum([a.GetTotalNumHs() for a in self.mol.GetAtoms()])
                cbh_rcts[cbh_level] = {'[H][H]':(pdt_H - rct_H)/2}
            else:
                # get the previous products + branch
                cbh_rcts[cbh_level] = deepcopy(cbh_pdts[cbh_level-1])
                if cbh_level in branch_rcts.keys():
                    cbh_rcts[cbh_level] = self.add_dicts(cbh_rcts[cbh_level], branch_rcts[cbh_level])
                    # cbh_rcts[cbh_level].update(branch_rcts[cbh_level])
            
            # simplify reaction equation
            all_residuals = list(cbh_pdts[cbh_level].keys()) + list(cbh_rcts[cbh_level].keys())
            mols_with_repeats = [i for i in list(set(all_residuals)) if all_residuals.count(i) > 1]
            for mol in mols_with_repeats:
                num_diff = cbh_pdts[cbh_level][mol] - cbh_rcts[cbh_level][mol]
                # more of this molecule in the products
                if num_diff > 0:
                    cbh_pdts[cbh_level][mol] = num_diff
                    del cbh_rcts[cbh_level][mol]
                # more of this molecule in the reactants
                elif num_diff < 0:
                    cbh_rcts[cbh_level][mol] = -1*num_diff
                    del cbh_pdts[cbh_level][mol]
                # equal number of this molecule on both sides
                elif num_diff == 0:
                    del cbh_pdts[cbh_level][mol]
                    del cbh_rcts[cbh_level][mol]

        return cbh_pdts, cbh_rcts
            

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
        All dictionaries have the form: {residual SMILE : num occurences}}

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


def main():
    cbh = buildCBH('CCC(F)(F)C(F)(F)C(C(O)=O)(F)(F)')
    for i in cbh.cbh_pdts:
        print(i, cbh.cbh_pdts[i])
    return cbh.cbh_pdts

if __name__ == '__main__':
    main()