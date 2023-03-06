from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons
import igraph
import numpy as np
from collections import defaultdict


class buildCBH:
    """
    Build the CBH scheme for a given SMILES string of a molecule. It gets the RDKit 
    molecule object and graph representations of the SMILES.

    ATTRIBUTES
    ----------
    :mol:           [RDKit Mol obj]
    :smiles:        [str] RDKit's SMILES respresentation of the target molecule
    :graph:         [IGraph graph obj] IGraph's graph representation of the target 
                                        molecule
    :graph_adj:     [np array] adjacency matrix of target molecule graph
    :graph_dist:    [np array] distance matrix between all pairs of nodes
    
    Same as above but with explicit hydrogens
    :mol_h:         [RDKit Mol obj] molecule 
    :smiles_h:      [str] RDKit's SMILES respresentation of the target molecule
    :graph_h:       [IGraph graph obj] IGraph's graph representation of the target 
                                        molecule
    :graph_adj_h:   [np array] adjacency matrix of target molecule graph
    :graph_dist_h:  [np array] distance matrix between all pairs of nodes

    :cbh_rcts:      [nested dict] The reactant side of each CBH level 
                                {cbh_level: {residual SMILES : num occurences}}
    :cbh_pdts:      [nested dict] The product side of each CBH level 
                                {cbh_level: {residual SMILES : num occurences}}

    :ignore_F2:     [bool] Whether to avoid using F2 when saturate='F'
    :highest_cbh:   [int] The highest possible rung for the molecule's CBH scheme.

    METHODS
    -------
    build_scheme    The workhorse of the class. Builds the CBH hierarchy of a given 
                    molecule
    atom_centric    [helper] computes the atom-centric CBH rungs
    bond_centric    [helper] computes the bond-centric CBH rungs
    CBH_0_F         [helper] constructs the fluorinated CBH-0 rung, but avoids the 
                    use of F2 which leads to a smaller number of precursor molecules
    count_repeats   [-static] Count the number of repeats in a list and returns a 
                    dictionary
    replace_implicit_Hs   [-static] Replace specified atoms with a desired atom
    visualize       Used to visualize CBH reactions in Jupyter Notebook. 
    """

    def __init__(self, smiles:str, saturate=1, allow_overshoot=False, ignore_F2=True):
        """
        Constructs the attributes of buildCBH class.

        ARGUMENTS
        ---------
        :smiles:             [str] 
                SMILES string represnting a molecule

        :saturate:          [int or str] (default=1 or 'H')
                The int or str representation of the default molecule that will 
                saturate the heavy atoms. Usually 'H' (1), but is also often 
                'F' (9) or 'Cl' (17). Currently it only supports halogens. 

        :allow_overshoot:   [bool] (default=False)
                Choose to allow a precursor to have a substructure of the target 
                species, as long as the explicit target species does not show up 
                on the product side. 
                    Ex) if species='CC(F)(F)F' and saturate=9, the highest CBH rung 
                        would be: 
                        False - CBH-0-F
                        True - CBH-1-F (even though C2F6 is generated as a precursor)

        :ignore_F2:         [bool] (default=True) 
                Avoid using fluorine gas (F2) when saturate='F'. 
                Only works when the given molecule contains a combination of only 
                the atoms: C, H, O, F.
        """
        self.mol = Chem.MolFromSmiles(smiles) # RDkit molecule object
        self.smiles = Chem.CanonSmiles(smiles) # rewrite SMILES str in standard forms

        self.graph = mol2graph(self.mol) # Graph representation of molecule
        self.graph_adj = np.array(self.graph.get_adjacency().data) # Graph Adjacency Matrix
        self.graph_dist = np.array(self.graph.shortest_paths()) # Matrix holding distances between vertices

        # Molecule attributes with explicit hydrogens
        self.mol_h = Chem.AddHs(self.mol)
        self.smiles_h = Chem.MolToSmiles(Chem.AddHs(self.mol_h))
        self.graph_h = mol2graph(self.mol_h)
        self.graph_adj_h = np.array(self.graph_h.get_adjacency().data)
        self.graph_dist_h = np.array(self.graph_h.shortest_paths())

        # Build CBH Scheme
        self.ignore_F2 = ignore_F2
        self.cbh_pdts, self.cbh_rcts = self.build_scheme(saturate=saturate, allow_overshoot=allow_overshoot)
        # Highest CBH rung
        self.highest_cbh = max(self.cbh_pdts.keys())


    def build_scheme(self, saturate=1, allow_overshoot=False) -> tuple:
        """
        Build CBH scheme and store the SMILES strings in dictionaries for each CBH 
        level.

        ARGUMENTS
        ---------
        :saturate:          [int or str] (default=1 or 'H') 
                                The integer or string representation of the default 
                                molecule that will saturate the heavy atoms. 
                Ex)'H' (1), but is also often 'F' (9) or 'Cl' (17).

        :allow_overshoot:   [bool] (default=False)
                                Choose to allow a precursor to have a substructure 
                                of the target species, as long as the explicit 
                                target species does not show up on the product 
                                side. 
                Ex) if species='CC(F)(F)F' and saturate=9, the highest CBH rung 
                    would be: 
                    False - CBH-0-F
                    True - CBH-1-F (even though C2F6 is generated as a precursor)

        RETURNS
        -------
        (cbh_pdts, cbh_rcts) [tuple]

            cbh_pdts    [nested dict] The product side of each CBH level 
                                {cbh_level: {residual SMILES : num occurences}}
            cbh_rcts    [nested dict] The reactant side of each CBH level 
                                {cbh_level: {residual SMILES : num occurences}}
        """

        # saturation atom parsing
        ptable = Chem.GetPeriodicTable()
        if type(saturate) == str:
            saturate = ptable.GetAtomicNumber(saturate)
        saturate_sym = ptable.GetElementSymbol(saturate)

        # all elements except for C H O and F
        atom_symbols = [ptable.GetElementSymbol(e) for e in range(118)]
        atom_symbols.remove('C')
        atom_symbols.remove('H')
        atom_symbols.remove('O')
        atom_symbols.remove('F')
        CBH_0_F_cond = True not in [True for a in atom_symbols if a in self.smiles_h]
        
        cbh_pdts = {} # CBH level products
        cbh_rcts = {} # CBH level reactants

        # "important_atoms" are just non-saturated atoms
        important_atoms = [atom for atom in self.mol_h.GetAtoms() if atom.GetAtomicNum() != saturate]
        important_idx = [atom.GetIdx() for atom in important_atoms]

        # Terminal atoms are those that only connected to one atom that is not the saturation atom
        # Indices of atoms where branching occurs or is a terminal 
        terminal_idx = []
        branch_idx = []
        branch_degrees = []
        # cycle through non-saturate atoms
        for atom in important_atoms:
            # get neighbors of the atom
            neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
            # get number of neighbors of the atom that are not saturate atoms
            non_saturate = len(neighbors) - neighbors.count(saturate)
            # branching
            if non_saturate > 2:
                branch_idx += [atom.GetIdx()]
                branch_degrees += [non_saturate]
            # terminal
            elif non_saturate == 1: 
                terminal_idx += [atom.GetIdx()]

        cbh_level = 0 # init
        trigger = False # trigger to break loop and end generation of next rung

        while True: # As long as the original molecule is not recreated in CBH levels
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
            
            # (i.e., C2F6 in product side for CH3CF3)
            if allow_overshoot:
                # End loop if the target molecules shows up on the product side, allowing overshooting
                if self.smiles in set(residuals+terminals):
                    trigger = True
            else:
                # End loop if the target molecules shows up on the product side
                # The substructure search removes the possibilty of "overshooting" or "circularity" 
                if True in [Chem.MolFromSmiles(r).HasSubstructMatch(self.mol) for r in set(residuals+terminals)]:
                    trigger = True

            if trigger:
                if saturate == 9 and CBH_0_F_cond and NumRadicalElectrons(self.mol)==0 and 'F' in self.smiles_h and 'C' in self.smiles_h and self.ignore_F2:
                    # if saturation is fluorine and the only elements present are C H O or F
                    cbh_pdts[0], cbh_rcts[0] = self.CBH_0_F()
                del cbh_rcts[cbh_level]
                if cbh_rcts[0]['[H][H]'] == 0:
                    del cbh_rcts[0]['[H][H]']
                break

            # Count num of each residual and return as dictionary
            cbh_rcts[cbh_level+1] = self.count_repeats(residuals)
            cbh_pdts[cbh_level] = self.count_repeats(residuals+terminals)

            # 2. CBH reactants
            if cbh_level == 0:
                # Get number of H in product (bakes in stoichiometry)
                pdt_H = sum([cbh_pdts[0][smiles]*Chem.MolFromSmiles(smiles).GetAtomWithIdx(0).GetTotalNumHs() \
                    for smiles in cbh_pdts[0].keys()])
                # Get number of H in target molecule
                rct_H = sum([a.GetTotalNumHs() for a in self.mol.GetAtoms()])
                cbh_rcts[0] = {'[H][H]':(pdt_H - rct_H)/2}
                
                # Count number of saturation atoms to balance
                if saturate != 1:
                    pdt_F = sum([cbh_pdts[0][smiles]*smiles.count(saturate_sym) for smiles in cbh_pdts[0].keys()])
                    rct_F = sum([cbh_rcts[0][smiles]*smiles.count(saturate_sym) for smiles in cbh_rcts[0].keys()])
                    rct_F += self.smiles.count(saturate_sym)
                    cbh_rcts[0][f'{saturate_sym}{saturate_sym}'] = (pdt_F - rct_F)/2
            else:
                # Get the previous products + branch
                if len(new_branches) != 0:
                    cbh_rcts[cbh_level] = add_dicts(cbh_rcts[cbh_level], self.count_repeats(new_branches))

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


    def atom_centric(self, dist:int, return_smile=True, atom_indices=[], saturate=1) -> list:
        """
        Returns a list of RDkit molecule objects that are the residuals species 
        produced by atom-centric CBH steps.

        ARGUMENTS
        ---------
        :dist:          [int] 
                The radius or number of atoms away from a target atom (0,1,...).

        :return_smile:  [bool] (default=True)
                Return SMILES string instead of RDKit.Chem Mol object.

        :atom_indices:  [list] (default=[]) 
                Specific atom indices to find atom-centric residuals.

        :saturate:      [int] (default=1)
                 The integer or string representation of the default molecule 
                 that will saturate the heavy atoms. Usually 1 (hydrogen), 
                 but is also often 9 (fluorine) or 17 (chloride).

        RETURNS
        -------
        :residuals: [list] A list of RDkit molecule objects or SMILES strings of 
                        residual species
        """
        residuals = [] # will hold residual species

        if len(atom_indices) != 0:
            idxs = atom_indices
        else: # compute subgraphs for all atoms in graph
            if saturate == 1:
                # Even though we use the graph with explicit H's in the for loop, 
                # the indices in graph_adj correspond exactly to the indices in the graph_adj_h case
                idxs = range(len(self.graph_adj))
            else:
                idxs = range(len(self.graph_adj_h))
        # cycle through relevant atoms
        for i in idxs:
            # Skip if a given index is a saturation atom
            if self.graph_h.vs()[i]['AtomicNum'] == saturate:
                continue
            # atom indices that are within the given radius
            atom_inds = np.where(self.graph_dist_h[i] <= dist)[0]
            # create rdkit mol objects from subgraph
            residual = graph2mol(self.graph_h.subgraph(atom_inds))
            if saturate != 1:
                # only get impl_valence for H or C since we only want to replace 
                # H's that are connected to these atoms
                # ie. we don't want to replace H on OH with F --> OF is a terrible prediction
                impl_valence = {atom.GetIdx() : atom.GetImplicitValence() for atom in residual.GetAtoms() \
                    if atom.GetImplicitValence() > 0 and atom.GetAtomicNum() in (1,6)}
                residual = self.replace_implicit_Hs(residual, impl_valence, saturate)

            if return_smile:
                # make explicit H's --> implicit
                residual = Chem.CanonSmiles(Chem.MolToSmiles(residual))
            residuals.append(residual)
        return residuals


    def bond_centric(self, dist:int, return_smile=True, saturate=1) -> list:
        """
        Returns a list of RDkit molecule objects that are the residuals species 
        produced by bond-centric CBH steps.

        ARGUMENTS
        ---------
        :dist:          [int] 
                The radius or number of bonds away from a target bond (0,1,...)

        :return_smile:  [bool] (default = True)
                Return SMILES string instead of RDKit.Chem Mol object

        :saturate:      [int] (default = 1 or 'H') 
                The integer or string representation of the default molecule that 
                will saturate the heavy atoms. Usually 'H' (1), but is also 
                often 'F' (9) or 'Cl' (17).

        RETURNS
        -------
        :residuals: [list] A list of RDkit molecule objects or SMILES strings of 
                            residual species
        """
        residuals = [] # will hold residual species
        dist += 1 # a specified distance of 0 implies the given edge --> this ensures this adjustment is made
        # cycle through edges
        for e in self.graph_h.es():
            if self.graph_h.vs()[e.source]['AtomicNum'] == saturate or self.graph_h.vs()[e.target]['AtomicNum'] == saturate:
                continue # skip edge if it's connected to a saturation atom
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
                # create rdkit mol objects from subgraph --> MUST be no explicit hydrogens (except for radicals)
                residual = graph2mol(self.graph_h.subgraph(edge_inds))
                if saturate != 1:
                    # {atom index : implicit valence} for carbon atoms that have more than 1 implicit hydrogen
                    #   (possibly too specific)
                    impl_valence = {atom.GetIdx() : atom.GetImplicitValence() for atom in residual.GetAtoms() \
                        if atom.GetImplicitValence() > 0 and atom.GetAtomicNum() == 6}
                    # replace implicit hydrogens with the saturation atom
                    residual = self.replace_implicit_Hs(residual, impl_valence, saturate)
                if return_smile:
                    # Remove explicit hydrogens from SMILES string
                    residual = Chem.CanonSmiles(Chem.MolToSmiles(residual))
                residuals.append(residual)
        return residuals


    def CBH_0_F(self) -> tuple:
        """
        Calculates an alternative isogyric fluorinated CBH-0 scheme that 
        avoids the use of F2.
        (a - 1/4c)*CH4 + b*H2O + c/4*CF4 + (d/2 - 2a - b + 1/2*c)*H2 --> CaObFcHd

        ARGUMENTS
        ---------
        :self:      [CBH.buildCBH obj]

        RETURNS
        -------
        (pdts, rcts)

        :pdts:  [dict] The product side of CBH-0-F \
                    {residual SMILES : num occurences}
        :rcts:  [dict] The reactant side of each CBH level \
                    {residual SMILES : num occurences}
        """
        
        atoms = ['C','O','F','H']
        coeffs = {a:self.smiles_h.count(a) for a in atoms}
        
        coeff_ch4 = coeffs['C'] - 1/4*coeffs['F']
        coeff_h2o = coeffs['O']
        coeff_cf4 = coeffs['F']/4
        coeff_h2 = -1 * (coeffs['H']/2 - 2*coeffs['C'] - coeffs['O'] + 1/2*coeffs['F'])

        pdts = {'C':coeff_ch4, 'O':coeff_h2o, 'FC(F)(F)F':coeff_cf4}
        rcts = {'[H][H]':coeff_h2}
        if pdts['O'] == 0:
            del pdts['O']
        return pdts, rcts


    @staticmethod
    def count_repeats(ls:list) -> dict:
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
    def replace_implicit_Hs(mol, impl_valence_dict:dict, change:int):
        """
        Replace the implicit hydrogens with a 'change' atom. 
        Assumes 'change' atom is a halogen.

        ARGUMENTS
        ---------
        :impl_valence_dict: [dict] {atom idx : implicit valence}
        :target:            [int] Element number of the atom to replace
        :change:            [int] Element number of the atom to replace with

        RETURNS
        -------
        :new_mol:   [RDkit Mol] Mol with replaced atoms
        """
        
        new_mol = Chem.RWMol(Chem.AddHs(mol))
        for i_atom, impl_val in impl_valence_dict.items():
            atom =  new_mol.GetAtomWithIdx(i_atom)
            keep_track_impl_val = 0
            for nbr in atom.GetNeighbors():
                # if neighbor is an implicit hydrogen
                if nbr.GetAtomicNum() == 1: 
                    nbr.SetAtomicNum(change)
                    keep_track_impl_val += 1
                    # stop replacing atoms once the num of replaced exceeds the num implicit valence
                    if keep_track_impl_val >= impl_val:
                        break
        Chem.SanitizeMol(new_mol)
        return new_mol

    
    def visualize(self, cbh_rung:int=[]):
        """
        Visualize the CBH scheme in a jupyter notebook. Can show all or specific 
        CBH rungs.

        ARGUMENTS
        ---------
        :cbh_rung:  [int] (default = [] --> all rungs) CBH rung to display. 
                        A value of '-1' can be used to get the highest rung.

        RETURNS
        -------
        None
        """

        if type(cbh_rung) != int:
            cbh_levels = self.cbh_pdts.keys()
        else:
            cbh_levels = list(self.cbh_pdts.keys())
            cbh_levels.sort()
            try:
                cbh_levels = [cbh_levels[cbh_rung]]
            except IndexError:
                print(f'CBH rung {cbh_rung} does not exist for {self.smiles}.\nThe highest CBH rung for this species is: {max(cbh_levels)}')
                return

        # cycle through each CBH rung
        for cbh_level in cbh_levels:
            self.__visualize(cbh_level)
            

    def __visualize(self, cbh_level:int):
        """
        Helper function that actually creates the images to visualize. 
        Designed for Jupyter Notebook.

        ARGUMENTS
        ---------
        :cbh_level:     [int] CBH rung to visualize
        """

        from rdkit.Chem import PandasTools
        from IPython.display import display
        from pandas import DataFrame, concat
        from rdkit.Chem.Draw import IPythonConsole
        IPythonConsole.drawOptions.addAtomIndices = False

        # Visualize reaction without atom indices
        # Create pandas df of reactant and products
        max_num_mols = max([len(v) for v in self.cbh_pdts.values()]+[len(v) for v in self.cbh_rcts.values()])

        rct_df = DataFrame(self.cbh_rcts[cbh_level].items(), columns=['smiles', 'num'])
        target = DataFrame({self.smiles:1}.items(), columns=['smiles', 'num'])
        rct_df = concat([target, rct_df[:]]).reset_index(drop = True)
        PandasTools.AddMoleculeColumnToFrame(rct_df, smilesCol='smiles')
        pdt_df = DataFrame(self.cbh_pdts[cbh_level].items(), columns=['smiles', 'num'])
        PandasTools.AddMoleculeColumnToFrame(pdt_df, smilesCol='smiles')
        
        print('\n-----------------------------------------------------------------------------------------------------\n')
        print(f'CBH RUNG {cbh_level}')
        print(f'\nReactants:')
        display(PandasTools.FrameToGridImage(rct_df, legendsCol="num", subImgSize=(200,120), molsPerRow=max_num_mols))
        print(f'\nProducts:')
        display(PandasTools.FrameToGridImage(pdt_df, legendsCol="num", subImgSize=(200,120), molsPerRow=max_num_mols))


def mol2graph(mol):
    """Molecule to Graph
    Converts a RDkit.Chem Mol object into an undirected igraph object.
    Sourced directly from: https://iwatobipen.wordpress.com/2018/05/30/active-learning-rdkit-chemoinformatics-ml/
    
    ARGUMENTS
    ---------
    :mol: RDkit.Chem Mol object - represents molecule
    
    RETURNS
    -------
    :g: [igraph Graph obj] - chemical graph of molecule attributes include: 
            atom idx, atom atomic numbers, atomic symbols, 
            bond start/end atom idx's, bond types
    """
    # Gather atom/bond attributes
    # atom attributes: atom number, atomic number, atom symbol
    # bond attributes: atom 1 index, atom 2 index, atom bond type as number
    atom_attributes = [(a.GetIdx(), a.GetAtomicNum(), a.GetSymbol(), a.GetNumRadicalElectrons()) for a in mol.GetAtoms()]
    bond_attributes = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType(), b.GetBondTypeAsDouble()) 
                       for b in mol.GetBonds()]
    # generate chemical graph
    g = igraph.Graph()
    # Create vertices for each Atom
    for a_attr in atom_attributes:
        g.add_vertex(a_attr[0], AtomicNum=a_attr[1], AtomicSymbol=a_attr[2], NumRad=a_attr[3])
    # Create edges for each Bond
    for b_attr in bond_attributes:
        g.add_edge(b_attr[0], b_attr[1], BondType=b_attr[2], BondTypeDouble=b_attr[3])
    return g 


def graph2mol(graph, return_smiles=False): 
    """Graph to Molecule
    Converts undirected igraph object to RDkit.Chem Mol object.
    Sourced directly from: https://iwatobipen.wordpress.com/2018/05/30/active-learning-rdkit-chemoinformatics-ml/
    
    ARGUMENTS
    ---------
    :graph:         [igraph graph obj]
            Chemical graph of molecule attributes include: 
                atom idx, atom atomic numbers, atomic symbols, 
                bond start/end atom idx's, bond types

    :return_smiles:  [bool] (default=False) 
            Return a SMILES string instead of a RDKit.Chem Mol object. 
            (uses Chem.CanonSmiles so implicit H's will be used if possible)
    
    RETURNS
    -------
    :mol:   Molecular representation as either a RDkit.Chem Mol object or SMILES string
    """
    mol = Chem.rdchem.RWMol()
    # Add each vertex as an Atom
    atom_num = 0 # counter
    for v in graph.vs():
        mol.AddAtom(Chem.Atom(v["AtomicNum"]))

        if v['NumRad'] != 0:
            # Set the num of radical electrons
            mol.GetAtomWithIdx(atom_num).SetNumRadicalElectrons(v['NumRad'])
            
        atom_num += 1
    # Add each edge as a Bond
    for e in graph.es():
        mol.AddBond(e.source, e.target, e['BondType'])
    mol = mol.GetMol()
    Chem.SanitizeMol(mol) # ensure implicit hydrogens are accounted
    
    # Generates SMILES str
    if return_smiles:
        # will always return SMILES string with implicit hydrogens
        mol = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    return mol


def add_dicts(*dictionaries: dict) -> dict:
    """
    Add the values within a dictionary together for matching keys.
    All dictionaries have the form: {residual SMILES : num occurences}}

    ARGUMENTS
    ---------
    :*dictionaries: [dict] dictionaries to add

    RETURNS
    -------
    :out_dict: [dict] the output dictionary where the number of matching 
                    SMILEs are added together
    """
    dd = defaultdict(list) # dictionary with elements of a list
    for d in [*dictionaries]:
        for key, value in d.items():
            # {key: [val1, val2]}
            dd[key].append(value)
    # add up the values for each key
    out_dict = {key:sum(dd[key]) for key in dd.keys()}
    # remove items where values = 0
    del_list = []
    for k, v in out_dict.items():
        if v == 0:
            del_list.append(k)
    for k in del_list:
        del out_dict[k]
    return out_dict
