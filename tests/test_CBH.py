#########################
# Test Suite for CBH.py #
#########################

import autocbh.CBH as CBH
from autocbh.CBH import mol2graph, graph2mol, add_dicts
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from pytest import raises
from rdkit.Chem import CanonSmiles

class TestMol2Graph:

    normal_smile = CanonSmiles('CC(=O)CC')

    adsorbate_single_bond = CanonSmiles('CC(=O)C[Pt]')
    adsorbate_double_bond = CanonSmiles('CC(=O)C=[Pt]')
    adsorbate_triple_bond = CanonSmiles('CC(=O)C#[Pt]')

    radical_singlet = CanonSmiles('CC(=O)C[CH2]')
    radical_doublet = CanonSmiles('CC(=O)C[CH]')
    radical_triplet = CanonSmiles('CC(=O)C[C]')

    def test_graph_radical_num(self):
        mol = MolFromSmiles(self.radical_triplet)
        g = mol2graph(mol)
        assert sum(g.vs.get_attribute_values('NumRad')) == 3
    
    def test_graph_radical_num_ads(self):
        mol = MolFromSmiles(self.adsorbate_single_bond)
        g = mol2graph(mol)
        assert sum(g.vs.get_attribute_values('NumRad')) == 0
    
    def test_graph_num_vtx(self):
        mol = MolFromSmiles(self.adsorbate_double_bond)
        g = mol2graph(mol)
        assert g.vcount() == 5
    
    def test_graph_num_vtx_explicitHs(self):
        mol = Chem.AddHs(MolFromSmiles(self.adsorbate_double_bond))
        g = mol2graph(mol)
        assert g.vcount() == 9

    def test_graph_num_edge(self):
        mol = MolFromSmiles(self.adsorbate_double_bond)
        g = mol2graph(mol)
        assert g.ecount() == 4
    
    def test_graph_num_edge_explicitHs(self):
        mol = Chem.AddHs(MolFromSmiles(self.adsorbate_double_bond))
        g = mol2graph(mol)
        assert g.ecount() == 8

    def test_graph_bondType_ads(self):
        mol = MolFromSmiles(self.adsorbate_triple_bond)
        elems = [a.GetSymbol() for a in mol.GetAtoms()]
        g = mol2graph(mol)
        pt_id = elems.index("Pt")
        edge_id_pt = [pt_id in edge_pair for edge_pair in g.get_edgelist()].index(True)
        pt_bondtype = g.es()['BondTypeDouble'][edge_id_pt]
        assert pt_bondtype == 3.0


class TestGraph2Mol:
    
    normal_smile = CanonSmiles('CC(=O)CC')

    adsorbate_single_bond = CanonSmiles('CC(=O)C[Pt]')
    adsorbate_double_bond = CanonSmiles('CC(=O)C=[Pt]')
    adsorbate_triple_bond = CanonSmiles('CC(=O)C#[Pt]')

    radical_singlet = CanonSmiles('CC(=O)C[CH2]')
    radical_doublet = CanonSmiles('CC(=O)C[CH]')
    radical_triplet = CanonSmiles('CC(=O)C[C]')

    rad_and_ads = CanonSmiles('[Pt]C(=O)C[CH]')

    def test_return_mol(self):
        mol = MolFromSmiles(self.adsorbate_triple_bond)
        g = mol2graph(mol)
        m = graph2mol(g)
        assert isinstance(m, Chem.rdchem.Mol)

    def test_return_canon_smile(self):
        mol = MolFromSmiles(self.adsorbate_triple_bond)
        smiles = Chem.CanonSmiles(self.adsorbate_triple_bond)
        g = mol2graph(mol)
        m_smiles = graph2mol(g, return_smiles=True)
        assert smiles == m_smiles

    def test_num_radicals(self):
        mol = MolFromSmiles(self.radical_triplet)
        g = mol2graph(mol)
        assert sum(g.vs()['NumRad']) == 3 # for triplet radical
    
    def test_set_radicals(self):
        mol = MolFromSmiles(self.radical_triplet)
        g = mol2graph(mol)
        ind_rad = g.vs()['NumRad'].index(3) # for triplet radical
        m = graph2mol(g)
        atom = m.GetAtomWithIdx(ind_rad)
        assert atom.GetNumRadicalElectrons() == 3

    def test_implicitHs(self):
        """To test Sanitize Mol"""
        mol = MolFromSmiles(self.rad_and_ads)
        g = mol2graph(mol)
        m = graph2mol(g)
        num_implicit = sum([atom.GetNumImplicitHs() for atom in m.GetAtoms()])
        assert num_implicit == 3
    
    def test_implicitHs_AddHs(self):
        """To test Sanitize Mol with AddHs"""
        mol = Chem.AddHs(MolFromSmiles(self.rad_and_ads))
        g = mol2graph(mol)
        m = graph2mol(g)
        num_implicit = sum([atom.GetNumImplicitHs() for atom in m.GetAtoms()])
        assert num_implicit == 0
    
    def test_set_radicals_addHs(self):
        mol = Chem.AddHs(MolFromSmiles(self.rad_and_ads))
        g = mol2graph(mol)
        m = graph2mol(g)
        smiles = Chem.MolToSmiles(m)
        assert smiles.count('H') == 3

    def test_return_smiles_AddHs(self):
        mol = Chem.AddHs(MolFromSmiles(self.rad_and_ads))
        g = mol2graph(mol)
        # smiles will be canon and only include implicit hydrogens
        # other than atoms with radicals
        smiles = graph2mol(g, return_smiles=True)
        assert smiles.count('H') == 1 


class TestAddDicts:

    dict1 = {'1': 2, '2': 5, 'hi': -4}
    dict2 = {'1': 3, '2':-3, 'hi': -2}
    dict3 = {'1':-2, '2': 2, 'hi': -4}
    dict4 = {'1':-2, '2':-5, 'hi': 4}
    dict5 = {'5':100, '2':-2}

    def test_add_dicts(self):
        d = add_dicts(self.dict1, self.dict2)
        assert d == {'1':5, '2':2, 'hi':-6}

    def test_add_dicts_with_sum_0(self):
        d = add_dicts(self.dict1, self.dict3)
        assert d == {'2':7, 'hi':-8}

    def test_add_dicts_with_all_sum_0(self):
        d = add_dicts(self.dict1, self.dict4)
        assert d == {}

    def test_add_multiple_dicts(self):
        d = add_dicts(self.dict1, self.dict2, self.dict5)
        assert d == {'1':5, 'hi':-6, '5':100}


class TestCBH0Ffunc:

    # will only be used if:
    #   sat = 9
    #   No radicals
    #   'F' in smiles_h
    #   'C' in smiles_h
    #   No elements other than C H O or F are in the smiles_h
    #       Ok if some are not present

    smiles3 = CanonSmiles('C(F)(F)(F)C(F)(F)F') # no H's

    def test_generic_pdts(self):
        smiles1 = CanonSmiles('CC(=O)C(F)(F)C') # C O F H
        cbh = CBH.buildCBH(smiles1)
        pdts, _ = cbh.CBH_0_F()
        true_pdt_dict = {CanonSmiles(k): v for k, v in {'C':3.5, 'O':1, 'FC(F)(F)F':0.5}.items()}
        assert pdts == true_pdt_dict
    
    def test_generic_rcts(self):
        smiles1 = CanonSmiles('CC(=O)C(F)(F)C') # C O F H
        cbh = CBH.buildCBH(smiles1)
        _, rcts = cbh.CBH_0_F()
        true_rct_dict = {CanonSmiles(k): v for k, v in {'[H][H]':5}.items()}
        assert rcts == true_rct_dict

    def test_no_oxy_pdts(self):
        smiles2 = 'CCC(F)(F)C' # no O's
        cbh = CBH.buildCBH(smiles2)
        pdts, _ = cbh.CBH_0_F()
        true_pdt_dict = {CanonSmiles(k): v for k, v in {'C':3.5, 'FC(F)(F)F':0.5}.items()}
        assert pdts == true_pdt_dict

    def test_no_oxy_rcts(self):
        smiles2 = 'CCC(F)(F)C' # no O's
        cbh = CBH.buildCBH(smiles2)
        _, rcts = cbh.CBH_0_F()
        true_rct_dict = {CanonSmiles(k): v for k, v in {'[H][H]':3}.items()}
        assert rcts == true_rct_dict
    
    def test_no_h_or_o_pdts(self):
        smiles3 = 'C(F)(F)(F)C(F)(F)F' # no H's, no O's
        cbh = CBH.buildCBH(smiles3)
        pdts, _ = cbh.CBH_0_F()
        true_pdt_dict = {CanonSmiles(k): v for k, v in {'C':0.5, 'FC(F)(F)F':1.5}.items()}
        assert pdts == true_pdt_dict

    def test_no_h_or_o_rcts(self):
        smiles3 = 'C(F)(F)(F)C(F)(F)F' # no H's, no O's
        cbh = CBH.buildCBH(smiles3)
        _, rcts = cbh.CBH_0_F()
        true_rct_dict = {CanonSmiles(k): v for k, v in {'[H][H]':1}.items()}
        assert rcts == true_rct_dict

    def test_no_h_pdts(self):
        smiles4 = 'C(F)(F)(F)C(F)=O' # no H's
        cbh = CBH.buildCBH(smiles4)
        pdts, _ = cbh.CBH_0_F()
        true_pdt_dict = {CanonSmiles(k): v for k, v in {'C':1, 'O':1, 'FC(F)(F)F':1}.items()}
        assert pdts == true_pdt_dict

    def test_no_h_rcts(self):
        smiles4 = 'C(F)(F)(F)C(F)=O' # no H's
        cbh = CBH.buildCBH(smiles4)
        _, rcts = cbh.CBH_0_F()
        true_rct_dict = {CanonSmiles(k): v for k, v in {'[H][H]':3}.items()}
        assert rcts == true_rct_dict


def test_count_repeats():
    ls = [CanonSmiles(k) for k in ['CC(F)C', 'C', 'C', 'FCF', 'FCF', 'FCF']]
    repeats = CBH.buildCBH.count_repeats(ls)
    true_repeats_dict = {CanonSmiles(k): v for k, v in {'CC(F)C':1, 'C':2, 'FCF':3}.items()}
    assert repeats == true_repeats_dict


class TestReplaceImplicitH:
    """
    Replacement protocol

    - Should replace the specified 'target' atoms with the 'change' atom

    - When called by atom_centric or bond_centric, those target atoms are chosen 
        by finding the implicit valence (implicit H's) on C that were inserted 
        when using graph2mol. Thus, replace_atoms will replace those implicit H's 
        (ie. saturation H's) with the desired saturated atom.

    - atom_centric and bond_centric generate fragments (residuals) that are automatically 
        saturated with H. So prior to replace_atom, impl_valence (implicit H's) are
        found for each atom and put into a dictionary with the corresponding atom.
        Then, with replace_atom, those implicit H's are replaced with the target atom.
        
        Initially, after the original molecule is loaded, fragments for CBH are created 
        from subgraphs that only involve nodes/atoms that are not saturation atoms. 
        Thus, the fragments are automatically saturated with implicit H during graph2mol 
        and have to be replaced with the saturation atom afterwards. In graph2mol, 
        Chem.SanitizeMol will add implicit H's to the rdkit Mol object, but keep 
        any explicit H's within the object explicit. Thus, atom.GetImplicitValence()
        will not include explicit H's [test case: TestGraph2Mol.test_implicitHs_AddHs()]

    - note that in bond centric, there is a condition to find impl_valence that is 
        atom.GetAtomicNum() in (1,6) --> The "1" part of this injecture will only 
        ever occur if the residual is H2 --> that will never happen

    - impl_valence dict should find 
    """
    
    def test_replace_H_w_Cl(self):
        """Make sure you replace the 6 H's with Cl so there is a total
        of 7 Cl's in the molecule"""
        rad_ads_F_Cl = '[Pt]C(=O)C[CH]CCC(O)(Cl)F'
        mol = MolFromSmiles(rad_ads_F_Cl)
        # Find carbons with implicit H's attached
        # only three carbon atoms where each have two implicit valences
        impl_valence = {atom.GetIdx() : atom.GetImplicitValence() for atom in mol.GetAtoms() \
                    if atom.GetImplicitValence() > 0 and atom.GetAtomicNum() == 6}
        new_mol = CBH.buildCBH._replace_implicit_Hs(mol, impl_valence, 17)
        smiles = Chem.MolToSmiles(new_mol)
        assert smiles.count('Cl') == 7
    
    def test_replace_H_w_Cl_specific(self):
        """Only replace 4 of the 6 implicit H's in [Pt]C(=O)C[CH]CCC(O)(Cl)F
        and make sure that the 4 added Cl's are added to the right carbons."""
        rad_ads_F_Cl = '[Pt]C(=O)C[CH]CCC(O)(Cl)F'
        mol = MolFromSmiles(rad_ads_F_Cl)
        # Find carbons with implicit H's attached
        # only three carbon atoms where each have two implicit valences
        impl_valence = {atom.GetIdx() : atom.GetImplicitValence() for atom in mol.GetAtoms() \
                    if atom.GetImplicitValence() > 0 and atom.GetAtomicNum() == 6}
        del impl_valence[list(impl_valence.keys())[1]]
        new_mol = CBH.buildCBH._replace_implicit_Hs(mol, impl_valence, 17)
        carbon_atom_inds = list(impl_valence.keys())
        new_Cl_atoms = [nbr.GetAtomicNum() for atom in carbon_atom_inds for nbr in new_mol.GetAtomWithIdx(atom).GetNeighbors() if nbr.GetAtomicNum() == 17]
        assert new_Cl_atoms.count(17) == 4


class TestAtomCentric:
    
    def test_indices_are_equal_H_and_nonH(self):
        """Check whether added H's are just appended to index by checking
        equivalency of {index:atom} between mol and mol_h.
        This helps test the case where no atom_indices are given"""
        smiles = '[Pt]C(=O)NSF' # only one added H in the middle of the molec
        mol = MolFromSmiles(smiles)
        mol_ind2atom = {v.index:v['AtomicNum'] for v in mol2graph(mol).vs()}
        mol_h = Chem.AddHs(mol)
        mol_h_ind2atom = {v.index:v['AtomicNum'] for v in mol2graph(mol_h).vs()}
        max_ind = max(mol_ind2atom.keys())
        assert {i:mol_h_ind2atom[i] for i in range(max_ind+1)} == mol_ind2atom

    def test_atom_inds(self):
        """Check if it computes the right residuals for a given set of indices"""
        smiles = '[Pt]C(=O)NSF'
        mol = MolFromSmiles(smiles)
        atom2ind = {v['AtomicNum']:v.index for v in mol2graph(mol).vs()}
        # {78: 0, 6: 1, 8: 2, 7: 3, 16: 4, 9: 5}
        cbh = CBH.buildCBH(smiles)
        # get the residuals containing 1 hop atoms for C, N, F atoms
        residuals = set(cbh.atom_centric(1, return_smile=True, atom_indices=[atom2ind[a] for a in [6, 9, 7]]))
        
        correct = [CanonSmiles(s) for s in ['C(=O)([Pt])N', 'CNS', 'SF']]
        assert set(correct) == residuals

    def test_atom_inds_sat(self):
        """Check if it computes the right residuals for a given set of indices for the saturated case"""
        smiles = '[Pt]C(=O)NSF'
        mol = Chem.AddHs(MolFromSmiles(smiles))
        atom2ind = {v['AtomicNum']:v.index for v in mol2graph(mol).vs()}
        # {78: 0, 6: 1, 8: 2, 7: 3, 16: 4, 9: 5, 1: 6}
        cbh = CBH.buildCBH(smiles)
        # get the residuals containing 1 hop atoms for C, N, F atoms
        residuals = set(cbh.atom_centric(1, return_smile=True, atom_indices=[atom2ind[a] for a in [1, 7, 16]], saturate=9))
        
        correct = [CanonSmiles(s) for s in ['N', 'NSF', 'C(F)(F)(F)NS']]
        assert set(correct) == residuals

    def test_no_atom_inds(self):
        """Check if it computes the right residuals when no explicit indices are given"""
        smiles = '[Pt]C(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(1, saturate=1)) 

        correct = [CanonSmiles(s) for s in ['C[Pt]', 'C(=O)(N)[Pt]', 'C=O', 'CNS', 'NSF', 'SF']]
        assert set(correct) == residuals

    def test_no_atom_inds_sat(self):
        """Check if it computes the right residuals when no explicit indices are given in the saturated case"""
        smiles = '[Pt]C(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(1, saturate=9)) 

        correct = [CanonSmiles(s) for s in ['C(F)(F)(F)[Pt]', 'C(=O)(N)[Pt]', 'C(F)(F)=O', 'C(F)(F)(F)NS', 'NSF', 'N']]
        assert set(correct) == residuals
    
    def test_0(self):
        """Check 0 hop"""
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(0, saturate=1)) 

        correct = [CanonSmiles(s) for s in ['Cl', 'C', 'O', 'N', 'S', 'F']]
        assert set(correct) == residuals

    def test_sat_0(self):
        """Check 0 hop with saturation"""
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(0, saturate=9)) 

        correct = [CanonSmiles(s) for s in ['Cl', 'C(F)(F)(F)F', 'O', 'N', 'S', 'F']]
        assert set(correct) == residuals

    def test_sat_0_noH(self):
        """Check 0 hop with saturation but without any H's"""
        smiles = 'ClC(=O)N(F)SF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(0, saturate=9)) 

        correct = [CanonSmiles(s) for s in ['Cl', 'C(F)(F)(F)F', 'O', 'N', 'S']]
        assert set(correct) == residuals

    def test_sat_10(self):
        """Check # hop > size of molec with saturation"""
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(10, saturate=9)) 

        correct = [CanonSmiles(s) for s in ['ClC(=O)NSF', 'ClC(=O)NSF', 'ClC(=O)NSF', 'ClC(=O)NSF', 'ClC(=O)NSF', 'ClC(=O)NSF']]
        assert set(correct) == residuals

    def test_sat_10_noH(self):
        """Check # hop > size of molec with saturation but without any H's"""
        smiles = 'ClC(=O)N(F)SF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(10, saturate=9)) 

        correct = [CanonSmiles(s) for s in ['ClC(=O)N(F)SF', 'ClC(=O)N(F)SF', 'ClC(=O)N(F)SF', 'ClC(=O)N(F)SF', 'ClC(=O)N(F)SF']]
        assert set(correct) == residuals
    
    def test_return_mol(self):
        smiles = 'ClC(=O)N(F)SF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.atom_centric(10, return_smile=False, saturate=9)) 

        assert all([isinstance(residual, (Chem.rdchem.RWMol, Chem.RWMol, Chem.rdchem.Mol)) for residual in residuals])


class TestBondCentric:

    def test_dont_include_saturate(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.bond_centric(0, True, 1))

        correct = [CanonSmiles(s) for s in ['CCl', 'C=O', 'CN', 'SN', 'SF']]
        assert set(correct) == residuals

    def test_saturation(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.bond_centric(0, True, 9))

        correct = [CanonSmiles(s) for s in ['C(F)(F)(F)Cl', 'C(F)(F)=O', 'C(F)(F)(F)N', 'SN', 'N']]
        assert set(correct) == residuals

    def test_return_mol(self):
        smiles = 'ClC(=O)N(F)SF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.bond_centric(10, return_smile=False, saturate=9)) 

        assert all([isinstance(residual, (Chem.rdchem.RWMol, Chem.RWMol, Chem.rdchem.Mol)) for residual in residuals])
    
    def test_dist_1(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.bond_centric(1, True, 1))

        correct = [CanonSmiles(s) for s in ['C(Cl)(=O)N', 'C(Cl)(=O)NS', 'CNSF', 'NSF']]
        assert set(correct) == residuals
    
    def test_dist_1_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.bond_centric(1, True, 9))

        correct = [CanonSmiles(s) for s in ['C(Cl)(=O)N', 'C(Cl)(=O)NS', 'C(F)(F)(F)NSF', 'C(F)(F)(F)NS']]
        assert set(correct) == residuals

    def test_dist_10(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.bond_centric(10, True, 1))

        correct = ['ClC(=O)NSF']*6
        correct = [CanonSmiles(s) for s in correct]
        assert set(correct) == residuals
    
    def test_dist_10_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles)
        residuals = set(cbh.bond_centric(10, True, 9))

        correct = ['ClC(=O)NSF']*6
        correct = [CanonSmiles(s) for s in correct]
        assert set(correct) == residuals


class TestBuildSchemeGeneral:
    smiles = 'ClC(=O)NSF'
    cbh = CBH.buildCBH(smiles)

    def test_general_highest_rung(self):
        assert self.cbh.highest_cbh == 3
    
    def test_general_cbh_rcts_r0(self):
        assert self.cbh.cbh_rcts[0] == {CanonSmiles('[H][H]'): 6.0}

    def test_general_cbh_rcts_r1(self):
        assert self.cbh.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'N': 1, 'S': 1, 'C': 2}.items()}
    
    def test_general_cbh_rcts_r2(self):
        assert self.cbh.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'CN': 1, 'NS': 1}.items()}

    def test_general_cbh_rcts_r3(self):
        assert self.cbh.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'CNS': 1}.items()}
    
    def test_general_cbh_pdts_r0(self):
        assert self.cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'O': 1, 'F': 1, 'Cl': 1, 'N': 1, 'S': 1, 'C': 1}.items()}

    def test_general_cbh_pdts_r1(self):
        assert self.cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CN': 1, 'CCl': 1, 'NS': 1, 'FS': 1, 'C=O': 1}.items()}

    def test_general_cbh_pdts_r2(self):
        assert self.cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'NSF': 1, 'NC(=O)Cl': 1, 'CNS': 1}.items()}

    def test_general_cbh_pdts_r3(self):
        assert self.cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'O=C(Cl)NS': 1, 'CNSF': 1}.items()}

    ####################
    #### SATURATION ####
    ####################  
    def test_general_highest_rung_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.highest_cbh == 3
    
    def test_general_cbh_rcts_r0_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_rcts[0] == {'[H][H]': 4.0, 'FF': 2.0}

    def test_general_cbh_rcts_r1_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_rcts[1] == {'FC(F)(F)F': 2, 'N': 1}
    
    def test_general_cbh_rcts_r2_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_rcts[2] == {'NS': 1, 'NC(F)(F)F': 1}

    def test_general_cbh_rcts_r3_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_rcts[3] == {'FC(F)(F)NS': 1}
    
    def test_general_cbh_pdts_r0_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_pdts[0] == {'O': 1, 'F': 1, 'Cl': 1, 'FC(F)(F)F': 1, 'N': 1, 'S': 1}

    def test_general_cbh_pdts_r1_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_pdts[1] == {'O=C(F)F': 1, 'NS': 1, 'FC(F)(F)Cl': 1, 'NC(F)(F)F': 1}

    def test_general_cbh_pdts_r2_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_pdts[2] == {'FC(F)(F)NS': 1, 'NSF': 1, 'NC(=O)Cl': 1}

    def test_general_cbh_pdts_r3_sat(self):
        smiles = 'ClC(=O)NSF'
        cbh = CBH.buildCBH(smiles, 9)
        assert cbh.cbh_pdts[3] == {'O=C(Cl)NS': 1, 'FSNC(F)(F)F': 1}

    def test_raise_error_molecule_too_small_for_CBH(self):
        smiles = 'C'
        with raises(KeyError):
            CBH.buildCBH(smiles)

    def test_raise_error_for_CBH_of_H2(self):
        smiles = '[H][H]'
        with raises(KeyError):
            CBH.buildCBH(smiles)


class TestBuildSchemeOvershoot:
    ####################
    #### OVERSHOOT ####
    ####################  
    smiles = 'CC(F)(F)Cl'
    cbh = CBH.buildCBH(smiles, allow_overshoot=True)
    cbh9 = CBH.buildCBH(smiles, 9, allow_overshoot=True)
    
    def test_overshoot_highest_rung(self):
        assert self.cbh.highest_cbh == 1
    
    def test_overshoot_cbh_rcts_r0(self):
        assert self.cbh.cbh_rcts[0] == {CanonSmiles('[H][H]'): 4.0}

    def test_overshoot_cbh_rcts_r1(self):
        assert self.cbh.cbh_rcts[1] == {CanonSmiles('C'): 3}
    
    def test_overshoot_cbh_pdts_r0(self):
        assert self.cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'Cl': 1, 'F': 2, 'C': 2}.items()}

    def test_overshoot_cbh_pdts_r1(self):
        assert self.cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CCl': 1, 'CC': 1, 'CF': 2}.items()}

    
    #### OVERSHOOT SATURATION ####
    def test_overshoot_highest_rung_sat(self):
        assert self.cbh9.highest_cbh == 2
    
    def test_overshoot_cbh_rcts_r0_sat(self):
        assert self.cbh9.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H]': 0.5, 'FF': 4.5}.items()}

    def test_overshoot_cbh_rcts_r1_sat(self):
        assert self.cbh9.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)F': 4}.items()}
    
    def test_overshoot_cbh_rcts_r2_sat(self):
        assert self.cbh9.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 1}.items()}

    def test_overshoot_cbh_pdts_r0_sat(self):
        assert self.cbh9.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'Cl': 1, 'F': 3, 'FC(F)(F)F': 2}.items()}

    def test_overshoot_cbh_pdts_r1_sat(self):
        assert self.cbh9.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 1, 'FC(F)(F)Cl': 1, 'FC(F)F': 3}.items()}

    def test_overshoot_cbh_pdts_r2_sat(self):
        assert self.cbh9.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'CC(F)(F)F': 1, 'FC(F)(F)C(F)(F)Cl': 1}.items()}


class TestBuildSchemeRing:
    ##############
    #### RING ####
    ##############
    smiles = 'O=C1CC(F)CN1'
    cbh = CBH.buildCBH(smiles)
    cbh_sat = CBH.buildCBH(smiles, 9)
    
    def test_ring_highest_rung(self):
        assert self.cbh.highest_cbh == 3
    
    def test_ring_cbh_rcts_r0(self):
        assert self.cbh.cbh_rcts[0] == {CanonSmiles('[H][H]'): 8.0}

    def test_ring_cbh_rcts_r1(self):
        assert self.cbh.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'N': 1, 'C': 6}.items()}
    
    def test_ring_cbh_rcts_r2(self):
        assert self.cbh.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'CN': 2, 'CC': 3}.items()}

    def test_ring_cbh_rcts_r3(self):
        assert self.cbh.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'CNC': 1, 'CC(N)=O': 1, 'CC(C)F': 1, 'CCC': 1, 'CCN': 1}.items()}
    
    def test_ring_cbh_pdts_r0(self):
        assert self.cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'O': 1, 'F': 1, 'N': 1, 'C': 4}.items()}

    def test_ring_cbh_pdts_r1(self):
        assert self.cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CN': 2, 'CC': 3, 'C=O': 1, 'CF': 1}.items()}

    def test_ring_cbh_pdts_r2(self):
        assert self.cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'CNC': 1, 'CC(N)=O': 1, 'CC(C)F': 1, 'CCC': 1, 'CCN': 1}.items()}

    def test_ring_cbh_pdts_r3(self):
        assert self.cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'CC(F)CN': 1, 'CNC(C)=O': 1, 'CCNC': 1, 'CCC(C)F': 1, 'CCC(N)=O': 1}.items()}

    #### RING SATURATION ####
    def test_ring_highest_rung_sat(self):
        assert self.cbh_sat.highest_cbh == 3
    
    def test_ring_cbh_rcts_r0_sat(self):
        assert self.cbh_sat.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H]': 2.5, 'FF': 10.5}.items()}

    def test_ring_cbh_rcts_r1_sat(self):
        assert self.cbh_sat.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'N': 1, 'FC(F)(F)F': 10}.items()}
    
    def test_ring_cbh_rcts_r2_sat(self):
        assert self.cbh_sat.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 3, 'NC(F)(F)F': 2}.items()}

    def test_ring_cbh_rcts_r3_sat(self):
        assert self.cbh_sat.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'NC(=O)C(F)(F)F': 1, 'FC(F)(F)CC(F)(F)F': 1, 'NCC(F)(F)F': 1, 'FC(C(F)(F)F)C(F)(F)F': 1, 'FC(F)(F)NC(F)(F)F': 1}.items()}
    
    def test_ring_cbh_pdts_r0_sat(self):
        assert self.cbh_sat.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'O': 1, 'F': 6, 'N': 1, 'FC(F)(F)F': 4}.items()}

    def test_ring_cbh_pdts_r1_sat(self):
        assert self.cbh_sat.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 3, 'O=C(F)F': 1, 'FC(F)F': 5, 'NC(F)(F)F': 2}.items()}

    def test_ring_cbh_pdts_r2_sat(self):
        assert self.cbh_sat.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'NC(=O)C(F)(F)F': 1, 'FC(F)(F)CC(F)(F)F': 1, 'NCC(F)(F)F': 1, 'FC(C(F)(F)F)C(F)(F)F': 1, 'FC(F)(F)NC(F)(F)F': 1}.items()}

    def test_ring_cbh_pdts_r3_sat(self):
        assert self.cbh_sat.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'O=C(NC(F)(F)F)C(F)(F)F': 1, 'FC(CC(F)(F)F)C(F)(F)F': 1, 'FC(F)(F)CNC(F)(F)F': 1, 'NCC(F)C(F)(F)F': 1, 'NC(=O)CC(F)(F)F': 1}.items()}


class TestBuildSchemeRadical:
    #################
    #### RADICAL ####
    #################
    smiles = '[C]C[C]C(F)(F)[CH2]'
    cbh = CBH.buildCBH(smiles)
    cbh_sat = CBH.buildCBH(smiles, 9)

    def test_rad_highest_rung(self):
        assert self.cbh.highest_cbh == 3
    
    def test_rad_cbh_rcts_r0(self):
        assert self.cbh.cbh_rcts[0] == {CanonSmiles('[H][H]'): 6.0}

    def test_rad_cbh_rcts_r1(self):
        assert self.cbh.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'[CH2]': 1, 'C': 4}.items()}
    
    def test_rad_cbh_rcts_r2(self):
        assert self.cbh.cbh_rcts[2] == {CanonSmiles('[CH]C'): 2}

    def test_rad_cbh_rcts_r3(self):
        assert self.cbh.cbh_rcts[3] == {CanonSmiles('C[C]C'): 1}
    
    def test_rad_cbh_pdts_r0(self):
        assert self.cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'F': 2, '[CH2]': 1, '[CH]': 1, '[CH3]': 1, 'C': 2}.items()}

    def test_rad_cbh_pdts_r1(self):
        assert self.cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'[CH]C': 2, '[CH2]C': 1, '[C]C': 1, 'CF': 2}.items()}

    def test_rad_cbh_pdts_r2(self):
        assert self.cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'[CH]C([CH2])(F)F': 1, '[C]C[CH]': 1, 'C[C]C': 1}.items()}

    def test_rad_cbh_pdts_r3(self):
        assert self.cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'[CH2]C(F)(F)[C]C': 1, '[C]C[C]C': 1}.items()}

    #### RADICAL SATURATION ####
    def test_rad_highest_rung_sat(self):
        assert self.cbh_sat.highest_cbh == 3
    
    def test_rad_cbh_rcts_r0_sat(self):
        assert self.cbh_sat.cbh_rcts[0] == {CanonSmiles('FF'): 8.0}

    def test_rad_cbh_rcts_r1_sat(self):
        assert self.cbh_sat.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'F[C]F': 1, 'F[C](F)F': 2, 'FC(F)(F)F': 4}.items()}
    
    def test_rad_cbh_rcts_r2_sat(self):
        assert self.cbh_sat.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'F[C]C(F)(F)F': 2, 'F[C](F)C(F)(F)F': 1}.items()}

    def test_rad_cbh_rcts_r3_sat(self):
        assert self.cbh_sat.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'FC(F)(F)[C]C(F)(F)F': 1, 'F[C]C(F)(F)[C](F)F': 1}.items()}
    
    def test_rad_cbh_pdts_r0_sat(self):
        assert self.cbh_sat.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'F[C]F': 1, 'F[C](F)F': 1, 'F': 4, 'FC(F)(F)F': 2, '[C]F': 1}.items()}

    def test_rad_cbh_pdts_r1_sat(self):
        assert self.cbh_sat.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'F[C]C(F)(F)F': 2, 'F[CH]F': 2, '[C]C(F)(F)F': 1, 'FC(F)F': 2, 'F[C](F)C(F)(F)F': 1}.items()}

    def test_rad_cbh_pdts_r2_sat(self):
        assert self.cbh_sat.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)[C]C(F)(F)F': 1, 'F[C]C(F)(F)[C](F)F': 1, '[CH2]C(F)(F)F': 1, '[C]C[C]F': 1}.items()}

    def test_rad_cbh_pdts_r3_sat(self):
        assert self.cbh_sat.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'[CH2]C(F)(F)[C]F': 1, '[C]C[C]C(F)(F)F': 1, 'F[C](F)C(F)(F)[C]C(F)(F)F': 1}.items()}


class TestBuildSchemeBranch:
    ################
    #### BRANCH ####
    ################
    smiles = 'CC(CF)C(F)(F)Cl'
    cbh = CBH.buildCBH(smiles)
    cbh_sat = CBH.buildCBH(smiles, 9)

    def test_branch_highest_rung(self):
        assert self.cbh.highest_cbh == 3
    
    def test_branch_cbh_rcts_r0(self):
        assert self.cbh.cbh_rcts[0] == {CanonSmiles('[H][H]'): 7.0}

    def test_branch_cbh_rcts_r1(self):
        assert self.cbh.cbh_rcts[1] == {CanonSmiles('C'): 6}
    
    def test_branch_cbh_rcts_r2(self):
        assert self.cbh.cbh_rcts[2] == {CanonSmiles('CC'): 2}

    def test_branch_cbh_rcts_r3(self):
        assert self.cbh.cbh_rcts[3] == {CanonSmiles('CC(C)C'): 1}
    
    def test_branch_cbh_pdts_r0(self):
        assert self.cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'Cl': 1, 'F': 3, 'C': 4}.items()}

    def test_branch_cbh_pdts_r1(self):
        assert self.cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CCl': 1, 'CC': 3, 'CF': 3}.items()}

    def test_branch_cbh_pdts_r2(self):
        assert self.cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'CC(C)C': 1, 'CCF': 1, 'CC(F)(F)Cl': 1}.items()}

    def test_branch_cbh_pdts_r3(self):
        assert self.cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'CC(C)C(F)(F)Cl': 1, 'CC(C)CF': 1}.items()}

    #### BRANCH SATURATION ####
    def test_branch_highest_rung_sat(self):
        assert self.cbh_sat.highest_cbh == 2
    
    def test_branch_cbh_rcts_r0_sat(self):
        assert self.cbh_sat.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H]': 0.5, 'FF': 9.5}.items()}

    def test_branch_cbh_rcts_r1_sat(self):
        assert self.cbh_sat.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)F': 9}.items()}
    
    def test_branch_cbh_rcts_r2_sat(self):
        assert self.cbh_sat.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 3}.items()}
    
    def test_branch_cbh_pdts_r0_sat(self):
        assert self.cbh_sat.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'Cl': 1, 'F': 6, 'FC(F)(F)F': 4}.items()}

    def test_branch_cbh_pdts_r1_sat(self):
        assert self.cbh_sat.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 3, 'FC(F)(F)Cl': 1, 'FC(F)F': 6}.items()}

    def test_branch_cbh_pdts_r2_sat(self):
        assert self.cbh_sat.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)Cl': 1, 'FC(F)(F)C(C(F)(F)F)C(F)(F)F': 1, 'CC(F)(F)F': 1, 'FCC(F)(F)F': 1}.items()}


class TestBuildSchemeIgnoreF2:
    ###################
    #### IGNORE F2 ####
    ###################
    smiles = 'CC(=O)C(F)O'
    cbh = CBH.buildCBH(smiles, 9)
    cbh_ignore_false = CBH.buildCBH(smiles, 9, ignore_F2=False)

    #### TRUE ####
    def test_igf2_true_highest_rung_sat(self):
        assert self.cbh.highest_cbh == 2
    
    def test_igf2_true_cbh_rcts_r0_sat(self):
        assert self.cbh.cbh_rcts[0] == {CanonSmiles('[H][H]'): 5.0}

    def test_igf2_true_cbh_rcts_r1_sat(self):
        assert self.cbh.cbh_rcts[1] == {CanonSmiles('FC(F)(F)F'): 7}
    
    def test_igf2_true_cbh_rcts_r2_sat(self):
        assert self.cbh.cbh_rcts[2] == {CanonSmiles('FC(F)(F)C(F)(F)F'): 2}
    
    def test_igf2_true_cbh_pdts_r0_sat(self):
        assert self.cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'C': 2.75, 'O': 2, 'FC(F)(F)F': 0.25}.items()}

    def test_igf2_true_cbh_pdts_r1_sat(self):
        assert self.cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 2, 'O=C(F)F': 1, 'OC(F)(F)F': 1, 'FC(F)F': 4}.items()}

    def test_igf2_true_cbh_pdts_r2_sat(self):
        assert self.cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'O=C(C(F)(F)F)C(F)(F)F': 1, 'CC(F)(F)F': 1, 'OC(F)C(F)(F)F': 1}.items()}

    #### FALSE ####
    def test_igf2_false_highest_rung_sat(self):
        assert self.cbh_ignore_false.highest_cbh == 2
    
    def test_igf2_false_cbh_rcts_r0_sat(self):
        assert self.cbh_ignore_false.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H]': 2.0, 'FF': 8.0}.items()}

    def test_igf2_false_cbh_rcts_r1_sat(self):
        assert self.cbh_ignore_false.cbh_rcts[1] == {CanonSmiles('FC(F)(F)F'): 7}
    
    def test_igf2_false_cbh_rcts_r2_sat(self):
        assert self.cbh_ignore_false.cbh_rcts[2] == {CanonSmiles('FC(F)(F)C(F)(F)F'): 2}
    
    def test_igf2_false_cbh_pdts_r0_sat(self):
        assert self.cbh_ignore_false.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'O': 2, 'F': 5, 'FC(F)(F)F': 3}.items()}

    def test_igf2_false_cbh_pdts_r1_sat(self):
        assert self.cbh_ignore_false.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F': 2, 'O=C(F)F': 1, 'OC(F)(F)F': 1, 'FC(F)F': 4}.items()}

    def test_igf2_false_cbh_pdts_r2_sat(self):
        assert self.cbh_ignore_false.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'O=C(C(F)(F)F)C(F)(F)F': 1, 'CC(F)(F)F': 1, 'OC(F)C(F)(F)F': 1}.items()}


class TestBuildSchemeAdsorbate:

    single_smiles = 'C(Cl)=C(F)C=[Pt]'
    single_cbh = CBH.buildCBH(single_smiles, saturate=1, allow_overshoot=True, ignore_F2=True, surface_smiles='[Pt]')
    single_cbh_sat = CBH.buildCBH(single_smiles, saturate=9, allow_overshoot=True, ignore_F2=True, surface_smiles='[Pt]')

    double_smiles = 'C(F)C(=[Pt])C#[Pt]'
    double_cbh = CBH.buildCBH(double_smiles, saturate=1, allow_overshoot=True, ignore_F2=True, surface_smiles='[Pt]')
    double_cbh_sat = CBH.buildCBH(double_smiles, saturate=9, allow_overshoot=True, ignore_F2=True, surface_smiles='[Pt]')

    triple_smiles = '[Pt]C(F)C(Cl)C(=[Pt])C#[Pt]'
    triple_cbh = CBH.buildCBH(triple_smiles, saturate=1, allow_overshoot=True, ignore_F2=True, surface_smiles='[Pt]')
    triple_cbh_sat = CBH.buildCBH(triple_smiles, saturate=9, allow_overshoot=True, ignore_F2=True, surface_smiles='[Pt]')
    
    ###################
    #### ADSORBATE ####
    ###################
    def test_single_ads_0_rcts(self):
        assert self.single_cbh.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H].[Pt]': 8}.items()}

    def test_single_ads_0_pdts(self):
        assert self.single_cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'F.[Pt]': 1, 'Cl.[Pt]': 1, 'C.[Pt]': 3, '[PtH]': 4}.items()}

    def test_single_ads_1_rcts(self):
        assert self.single_cbh.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'C.[Pt]': 4}.items()}

    def test_single_ads_1_pdts(self):
        assert self.single_cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CCl.[Pt]': 1, 'CF.[Pt]': 1, 'CC.[Pt]': 1, 'C=[Pt]': 1, 'C=C.[Pt]': 1}.items()}

    def test_single_ads_2_rcts(self):
        assert self.single_cbh.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'CC.[Pt]': 1, 'C=C.[Pt]': 1}.items()}

    def test_single_ads_2_pdts(self):
        assert self.single_cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'CC=[Pt]': 1, 'C=CCl.[Pt]': 1, 'C=C(C)F.[Pt]': 1}.items()}

    def test_single_ads_3_rcts(self):
        assert self.single_cbh.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'C=C(C)F.[Pt]': 1}.items()}

    def test_single_ads_3_pdts(self):
        assert self.single_cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'CC(F)=CCl.[Pt]': 1, 'C=C(F)C=[Pt]': 1}.items()}

    def test_single_ads_sat_0_rcts(self):
        assert self.single_cbh_sat.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H].[Pt]': 0.5, 'FF.[Pt]': 8.5}.items()}

    def test_single_ads_sat_0_pdts(self):
        assert self.single_cbh_sat.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'FC(F)(F)F.[Pt]': 3, 'Cl.[Pt]': 1, 'F.[Pt]': 2, 'F[Pt]': 4.0}.items()}

    def test_single_ads_sat_1_rcts(self):
        assert self.single_cbh_sat.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)F.[Pt]': 5}.items()}

    def test_single_ads_sat_1_pdts(self):
        assert self.single_cbh_sat.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)=C(F)F.[Pt]': 1, 'FC(F)F.[Pt]': 2, 'FC(F)=[Pt]': 1, 'FC(F)(F)C(F)(F)F.[Pt]': 1, 'FC(F)(F)Cl.[Pt]': 1}.items()}

    def test_single_ads_sat_2_rcts(self):
        assert self.single_cbh_sat.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'FC(F)=C(F)F.[Pt]': 1, 'FC(F)(F)C(F)(F)F.[Pt]': 1}.items()}

    def test_single_ads_sat_2_pdts(self):
        assert self.single_cbh_sat.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'FC(F)=CCl.[Pt]': 1, 'FC(F)(F)C=[Pt]': 1, 'FC(F)=C(F)C(F)(F)F.[Pt]': 1}.items()}

    def test_single_ads_sat_3_rcts(self):
        assert self.single_cbh_sat.cbh_rcts[3] == {CanonSmiles('FC(F)=C(F)C(F)(F)F.[Pt]'): 1}

    def test_single_ads_sat_3_pdts(self):
        assert self.single_cbh_sat.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'FC(=CCl)C(F)(F)F.[Pt]': 1, 'FC(F)=C(F)C=[Pt]': 1}.items()}

    def test_double_ads_0_rcts(self):
        assert self.double_cbh.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H].[Pt]': 8}.items()}

    def test_double_ads_0_pdts(self):
        assert self.double_cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'F.[Pt]': 1, 'C.[Pt]': 3, '[PtH]': 5}.items()}

    def test_double_ads_sat_0_rcts(self):
        assert self.double_cbh_sat.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'FF.[Pt]': 9}.items()}

    def test_double_ads_sat_0_pdts(self):
        assert self.double_cbh_sat.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'F.[Pt]': 2, 'FC(F)(F)F.[Pt]': 3, 'F[Pt]': 5}.items()}

    def test_double_ads_1_rcts(self):
        assert self.double_cbh.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'C.[Pt]': 4}.items()}

    def test_double_ads_1_pdts(self):
        assert self.double_cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CF.[Pt]': 1, 'CC.[Pt]': 2, 'C=[Pt]': 1, 'C#[Pt]': 1}.items()}

    def test_double_ads_sat_1_rcts(self):
        assert self.double_cbh_sat.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)F.[Pt]': 5}.items()}

    def test_double_ads_sat_1_pdts(self):
        assert self.double_cbh_sat.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F.[Pt]': 2, 'FC(F)=[Pt]': 1, 'FC#[Pt]': 1, 'FC(F)F.[Pt]': 2}.items()}

    def test_double_ads_2_rcts(self):
        assert self.double_cbh.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'CC.[Pt]': 2}.items()}

    def test_double_ads_2_pdts(self):
        assert self.double_cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'CCF.[Pt]': 1, 'CC(C)=[Pt]': 1, 'CC#[Pt]': 1}.items()}

    def test_double_ads_sat_2_rcts(self):
        assert self.double_cbh_sat.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F.[Pt]': 2}.items()}

    def test_double_ads_sat_2_pdts(self):
        assert self.double_cbh_sat.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(=[Pt])C(F)(F)F': 1, 'FC(F)(F)C#[Pt]': 1, 'FCC(F)(F)F.[Pt]': 1}.items()}

    def test_double_ads_3_rcts(self):
        assert self.double_cbh.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'CC(C)=[Pt]': 1}.items()}

    def test_double_ads_3_pdts(self):
        assert self.double_cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'CC(=[Pt])CF': 1, 'CC(=[Pt])C#[Pt]': 1}.items()}

    def test_double_ads_sat_3_rcts(self):
        assert self.double_cbh_sat.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(=[Pt])C(F)(F)F': 1}.items()}

    def test_double_ads_sat_3_pdts(self):
        assert self.double_cbh_sat.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'FCC(=[Pt])C(F)(F)F': 1, 'FC(F)(F)C(=[Pt])C#[Pt]': 1}.items()}

    def test_triple_ads_0_rcts(self):
        assert self.triple_cbh.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H].[Pt]': 11}.items()}

    def test_triple_ads_0_pdts(self):
        assert self.triple_cbh.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'C.[Pt]': 4, 'Cl.[Pt]': 1, 'F.[Pt]': 1, '[PtH]': 6}.items()}

    def test_triple_ads_1_rcts(self):
        assert self.triple_cbh.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'C.[Pt]': 7}.items()}

    def test_triple_ads_1_pdts(self):
        assert self.triple_cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CC.[Pt]': 3, 'CF.[Pt]': 1, 'CCl.[Pt]': 1, 'C=[Pt]': 1, 'C#[Pt]': 1, 'C[Pt]': 1}.items()}

    def test_triple_ads_2_rcts(self):
        assert self.triple_cbh.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'CC.[Pt]': 3}.items()}

    def test_triple_ads_2_pdts(self):
        assert self.triple_cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'CC(C)=[Pt]': 1, 'CC(C)Cl.[Pt]': 1, 'CC(F)[Pt]': 1, 'CC#[Pt]': 1}.items()}

    def test_triple_ads_3_rcts(self):
        assert self.triple_cbh.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'CC(C)=[Pt]': 1, 'CC(C)Cl.[Pt]': 1}.items()}

    def test_triple_ads_3_pdts(self):
        assert self.triple_cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'CC(=[Pt])C#[Pt]': 1, 'CC(=[Pt])C(C)Cl': 1, 'CC(Cl)C(F)[Pt]': 1}.items()}

    def test_triple_ads_4_rcts(self):
        assert self.triple_cbh.cbh_rcts[4] == {CanonSmiles(k): v for k, v in {'CC(=[Pt])C(C)Cl': 1}.items()}

    def test_triple_ads_4_pdts(self):
        assert self.triple_cbh.cbh_pdts[4] == {CanonSmiles(k): v for k, v in {'CC(=[Pt])C(Cl)C(F)[Pt]': 1, 'CC(Cl)C(=[Pt])C#[Pt]': 1}.items()}

    def test_triple_ads_0_sat_rcts(self):
        assert self.triple_cbh_sat.cbh_rcts[0] == {CanonSmiles(k): v for k, v in {'[H][H].[Pt]': 0.5, 'FF.[Pt]': 11.5}.items()}

    def test_triple_ads_0_sat_pdts(self):
        assert self.triple_cbh_sat.cbh_pdts[0] == {CanonSmiles(k): v for k, v in {'FC(F)(F)F.[Pt]': 4, 'Cl.[Pt]': 1, 'F.[Pt]': 2, 'F[Pt]': 6.0}.items()}

    def test_triple_ads_1_sat_rcts(self):
        assert self.triple_cbh_sat.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)F.[Pt]': 8}.items()}

    def test_triple_ads_1_sat_pdts(self):
        assert self.triple_cbh_sat.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC#[Pt]': 1, 'FC(F)F.[Pt]': 2, 'FC(F)(F)[Pt]': 1, 'FC(F)=[Pt]': 1, 'FC(F)(F)C(F)(F)F.[Pt]': 3, 'FC(F)(F)Cl.[Pt]': 1}.items()}

    def test_triple_ads_2_sat_rcts(self):
        assert self.triple_cbh_sat.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(F)(F)F.[Pt]': 3}.items()}

    def test_triple_ads_2_sat_pdts(self):
        assert self.triple_cbh_sat.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(Cl)C(F)(F)F.[Pt]': 1, 'FC(F)(F)C#[Pt]': 1, 'FC([Pt])C(F)(F)F': 1, 'FC(F)(F)C(=[Pt])C(F)(F)F': 1}.items()}

    def test_triple_ads_3_sat_rcts(self):
        assert self.triple_cbh_sat.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(Cl)C(F)(F)F.[Pt]': 1, 'FC(F)(F)C(=[Pt])C(F)(F)F': 1}.items()}

    def test_triple_ads_3_sat_pdts(self):
        assert self.triple_cbh_sat.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'FC([Pt])C(Cl)C(F)(F)F': 1, 'FC(F)(F)C(=[Pt])C(Cl)C(F)(F)F': 1, 'FC(F)(F)C(=[Pt])C#[Pt]': 1}.items()}

    def test_triple_ads_4_sat_rcts(self):
        assert self.triple_cbh_sat.cbh_rcts[4] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(=[Pt])C(Cl)C(F)(F)F': 1}.items()}

    def test_triple_ads_4_sat_pdts(self):
        assert self.triple_cbh_sat.cbh_pdts[4] == {CanonSmiles(k): v for k, v in {'FC(F)(F)C(Cl)C(=[Pt])C#[Pt]': 1, 'FC([Pt])C(Cl)C(=[Pt])C(F)(F)F': 1}.items()}
    
    # test errors
    def test_invalid_surface_smiles(self):
        with raises(Exception):
            CBH.buildCBH('C', surface_smiles='Pt')
    
    def test_surface_is_not_single_element(self):
        with raises(Exception):
            CBH.buildCBH('C', surface_smiles='[Pt][Co]')

    # test what happens when you provide surf smiles but actual molecule doesn't have a surf species


class TestBuildSchemePhysiosorbed:
    smiles = 'ClC(=O)NSF.[Pt]'
    cbh = CBH.buildCBH(smiles, saturate=1, allow_overshoot=True, ignore_F2=False, surface_smiles='[Pt]')
    cbh_sat = CBH.buildCBH(smiles, saturate=9, allow_overshoot=True, ignore_F2=False, surface_smiles='[Pt]')

    ######################
    #### PHYSIOSORBED ####
    ######################

    def test_physiosorbed_but_surface_not_correct(self):
        with raises(Exception):
            CBH.buildCBH('C.[Pt]', surface_smiles='[Pt][Co]')
    
    def test_too_many_components(self):
        with raises(Exception):
            CBH.buildCBH('C.Cl.[Pt]', surface_smiles='[Pt]')
    
    def test_physiosorbed_without_surface(self):
        with raises(Exception):
            CBH.buildCBH('C.[Pt]')

    # Biggest issue: 
    # it is impossible to balance CBH-0 for molecules with double/triple bonds 
    # attached to carbons.
    def test_physio_1_rct(self):
        assert self.cbh.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'N.[Pt]': 1, 'C.[Pt]': 2, 'S.[Pt]': 1}.items()}
        
    def test_physio_1_pdt(self):
        assert self.cbh.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'CN.[Pt]': 1, 'NS.[Pt]': 1, 'FS.[Pt]': 1, 'C=O.[Pt]': 1, 'CCl.[Pt]': 1}.items()}

    def test_physio_2_rct(self):
        assert self.cbh.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'CN.[Pt]': 1, 'NS.[Pt]': 1}.items()}
        
    def test_physio_2_pdt(self):
        assert self.cbh.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'NC(=O)Cl.[Pt]': 1, 'CNS.[Pt]': 1, 'NSF.[Pt]': 1}.items()}

    def test_physio_3_rct(self):
        assert self.cbh.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'CNS.[Pt]': 1}.items()}
        
    def test_physio_3_pdt(self):
        assert self.cbh.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'CNSF.[Pt]': 1, 'O=C(Cl)NS.[Pt]': 1}.items()}

    
    def test_physio_1_rct_sat(self):
        assert self.cbh_sat.cbh_rcts[1] == {CanonSmiles(k): v for k, v in {'N.[Pt]': 1, 'FC(F)(F)F.[Pt]': 2}.items()}
        
    def test_physio_1_pdt_sat(self):
        assert self.cbh_sat.cbh_pdts[1] == {CanonSmiles(k): v for k, v in {'FC(F)(F)Cl.[Pt]': 1, 'NC(F)(F)F.[Pt]': 1, 'O=C(F)F.[Pt]': 1, 'NS.[Pt]': 1}.items()}

    def test_physio_2_rct_sat(self):
        assert self.cbh_sat.cbh_rcts[2] == {CanonSmiles(k): v for k, v in {'NC(F)(F)F.[Pt]': 1, 'NS.[Pt]': 1}.items()}
        
    def test_physio_2_pdt_sat(self):
        assert self.cbh_sat.cbh_pdts[2] == {CanonSmiles(k): v for k, v in {'FC(F)(F)NS.[Pt]': 1, 'NC(=O)Cl.[Pt]': 1, 'NSF.[Pt]': 1}.items()}

    def test_physio_3_rct_sat(self):
        assert self.cbh_sat.cbh_rcts[3] == {CanonSmiles(k): v for k, v in {'FC(F)(F)NS.[Pt]': 1}.items()}
        
    def test_physio_3_pdt_sat(self):
        assert self.cbh_sat.cbh_pdts[3] == {CanonSmiles(k): v for k, v in {'FSNC(F)(F)F.[Pt]': 1, 'O=C(Cl)NS.[Pt]': 1}.items()}

    