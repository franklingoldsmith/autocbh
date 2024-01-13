""" Test suite module for calcCBH"""
#############################
# Test Suite for calcCBH.py #
#############################

import os
from pytest import raises
from numpy import nan
import numpy as np
from rdkit.Chem import CanonSmiles
from autocbh.calcCBH import calcCBH, add_dicts
from autocbh.CBH import buildCBH

class TestInit:
    ranking_file = 'tests/dummy_data/rankings.yaml'
    methods_keys_file = 'tests/dummy_data/methods_keys.yaml'
    c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=ranking_file,
                method_keys_path=methods_keys_file)

    def test_raise_error_if_data_not_provided(self):
        with raises(ValueError):
            calcCBH(dataframe_path=None, force_generate_database=None)

    def test_default_rankings_path_is_valid(self):
        file = 'data/rankings.yaml'
        assert os.path.isfile(file)

    def test_default_methods_keys_path_is_valid(self):
        file = 'data/methods_keys.yaml'
        assert os.path.isfile(file)

    def test_simplified_method_keys_dict_attribute(self):
        for k, v in self.c.methods_keys_dict.items():
            self.c.methods_keys_dict[k] = set(v)
        assert self.c.methods_keys_dict == {'beef_vdw': set(['beef_vdw_E0', 'beef_vdw_zpe'])}

    def test_simplified_method_keys_attribute(self):
        assert set(self.c.methods_keys) == set(['beef_vdw_E0', 'beef_vdw_zpe'])

    def test_simplified_methods_attribute(self):
        assert set(self.c.methods) == set(['beef_vdw'])

    def test_simplified_rankings_attribute(self):
        assert self.c.rankings == {1: ['ATcT', 'exp'], 8: ['beef_vdw']}

    def test_simplified_rankings_rev_attribute(self):
        assert self.c.rankings_rev == {'ATcT':1, 'exp': 1, 'beef_vdw': 8}

    # no tests for: self.alternative_rxn, zero_out_heats

class TestCalc_Hf:
    ranking_file = 'tests/dummy_data/rankings.yaml'
    methods_keys_file = 'tests/dummy_data/methods_keys.yaml'
    c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=ranking_file,
                method_keys_path=methods_keys_file)

    def test_alt_rxn_option_input_str(self):
        with raises(Exception):
            self.c.calc_Hf(alt_rxn_option=1)

    def test_alt_rxn_option_input_not_in_list(self):
        with raises(Exception):
            self.c.calc_Hf(alt_rxn_option='fake option')

    def test_priority_option_input_str(self):
        with raises(Exception):
            self.c.calc_Hf(priority=1)

    def test_priority_input_not_in_list(self):
        with raises(Exception):
            self.c.calc_Hf(priority='fake option')

    def test_saturate_arg_is_list_or_tuple(self):
        with raises(Exception):
            self.c.calc_Hf(saturate=1)

    def test_saturate_arg_is_str_or_int(self):
        with raises(Exception):
            self.c.calc_Hf(saturate=[1.])

    def test_saturate_arg_str_in_ptable(self):
        with raises(Exception):
            self.c.calc_Hf(saturate=['not an element'])

    def test_saturate_arg_int_in_ptable(self):
        with raises(Exception):
            self.c.calc_Hf(saturate=[10000000])

    def test_surface_smiles_attribute_is_none(self):
        c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=self.ranking_file,
                method_keys_path=self.methods_keys_file)
        c.calc_Hf(surface_smiles='[Pt]')
        assert c.surface_smiles == '[Pt]'

    def test_hrxn_fcns_bad_key(self):
        with raises(Exception):
            # 'hi' doesn't exist in c.rankings or c.methods_keys
            self.c.calc_Hf(hrxn_fcns={'hi':CanonSmiles}) # random function

class TestWeight:
    ranking_file = 'tests/dummy_data/rankings.yaml'
    methods_keys_file = 'tests/dummy_data/methods_keys.yaml'
    c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=ranking_file,
                method_keys_path=methods_keys_file)

    def test_weight_reasonable_Hrxn(self):
        weights = self.c._weight(5., -15.)
        denom = 1/5 + 1/15
        assert weights == [1/5/denom, 1/15/denom]

    def test_weight_one_zero(self):
        weights = self.c._weight(5., 3., 0., 10.)
        assert weights == [0., 0., 1., 0.]

    def test_weight_mult_zeros(self):
        weights = self.c._weight(0., 3., 0., 0.)
        assert weights == [1/3, 0, 1/3, 1/3]

    def test_weight_nan(self):
        weights = self.c._weight(5., nan)
        assert weights == [1., 0.]

    def test_weght_nan_zero(self):
        weights = self.c._weight(5., 0, nan)
        assert weights == [0., 1., 0.]

def convert_dict_keys_to_CanonSmiles(dictionary:dict):
    """Assumes dictionary: {smiles: any}
    Returns the same dictionary but with standardized SMILES"""
    return {CanonSmiles(s):v for s, v in dictionary.items()}

def stoichiometric_math(energies_df, full_rxn, species, columns, convert_from_hartree_to_kjmol=False):
    ans = -1*np.sum(np.vstack((full_rxn[s]*energies_df.loc[s, columns] for s in species)))
    if convert_from_hartree_to_kjmol:
        hartree_to_kJpermole = 2625.499748
        ans *= hartree_to_kJpermole
    return ans

class TestHf:

    ranking_file = 'tests/dummy_data/rankings.yaml'
    methods_keys_file = 'tests/dummy_data/methods_keys.yaml'
    c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=ranking_file,
                method_keys_path=methods_keys_file)

    species = [CanonSmiles(s) for s in ['OC[Pt]', 'C.[Pt]', 'CO.[Pt]', 'C[Pt]']]
    keys = ['beef_vdw_E0', 'beef_vdw_zpe', 'DfH', 'DrxnH', 'source']
    methods_keys = ['beef_vdw_E0', 'beef_vdw_zpe']
    c.methods_keys = methods_keys
    c.methods_keys_dict = {'beef_vdw': methods_keys}

    c.energies.loc[CanonSmiles('OC[Pt]'), keys] = [-909.715635, 0.041159, 0., 0., nan]
    c.energies.loc[CanonSmiles('C.[Pt]'), keys] = [-889.101433, 0.044577, -81.3, 0., 'exp']
    c.energies.loc[CanonSmiles('CO.[Pt]'), keys] = [-910.330881, 0.052809, -245.0, 0., 'exp']
    c.energies.loc[CanonSmiles('C[Pt]'), keys] = [-888.477198, 0.036051, -47.2, 0., 'exp']

    cbh1_rxn = {CanonSmiles(s):coeff for s, coeff in {'OC[Pt]':-1, 'C.[Pt]':-1, 'CO.[Pt]':1, 'C[Pt]':1}.items()}

    def test_Hrxn_Hf_generic_sum_Hrxn(self):
        # OC[Pt]
        hrxn_true = stoichiometric_math(energies_df=self.c.energies, full_rxn=self.cbh1_rxn,
                                        species=self.species, columns=self.methods_keys,
                                        convert_from_hartree_to_kjmol=True)
        hf_partial_calc = stoichiometric_math(energies_df=self.c.energies,
                                              full_rxn=self.cbh1_rxn, species=self.species,
                                              columns='DfH', convert_from_hartree_to_kjmol=False)
        hf_true = [np.round(v - hf_partial_calc, 8) for v in [hrxn_true, hf_partial_calc]]

        hrxn, hf = self.c.Hf(s=CanonSmiles('OC[Pt]'), rxn=self.cbh1_rxn, skip_precursor_check=True, hrxn_fcns={})
        hrxn = {k:np.round(v,8) for k, v in hrxn.items()}
        hf = {k:np.round(v,8) for k, v in hf.items()}
        assert hrxn == {'beef_vdw':np.round(hrxn_true,8), 'ref':np.round(hf_partial_calc,8)} and hf == {'beef_vdw':hf_true[0], 'ref':hf_true[1]}

    def test_Hrxn_Hf_nan_in_energies_generic_sum_Hrxn(self):
        # OC[Pt]
        self.c.energies.loc[CanonSmiles('CO.[Pt]'), self.keys] = [-910.330881, nan, -245.0, 0., 'exp']
        
        hrxn_true = stoichiometric_math(energies_df=self.c.energies, full_rxn=self.cbh1_rxn,
                                        species=self.species, columns=self.methods_keys,
                                        convert_from_hartree_to_kjmol=True)
        hf_partial_calc = stoichiometric_math(energies_df=self.c.energies, full_rxn=self.cbh1_rxn,
                                              species=self.species, columns='DfH',
                                              convert_from_hartree_to_kjmol=False)
        hf_true = [np.round(v - hf_partial_calc, 8) for v in [hrxn_true, hf_partial_calc]]

        hrxn, hf = self.c.Hf(s=CanonSmiles('OC[Pt]'), rxn=self.cbh1_rxn, skip_precursor_check=True, hrxn_fcns={})
        hrxn = {k:np.round(v,8) for k, v in hrxn.items()}
        hf = {k:np.round(v,8) for k, v in hf.items()}
        assert hrxn['ref'] == np.round(hf_partial_calc,8) and np.isnan(hrxn['beef_vdw']) and hf['ref'] == hf_true[1] and np.isnan(hf['beef_vdw'])

    def test_custom_Hrxn_fcn(self):
        new_keys = ['custom1', 'custom2']
        self.methods_keys.extend(new_keys)

        self.c.methods_keys = self.methods_keys
        self.c.methods_keys_dict['custom'] = new_keys
        self.c.methods_keys_dict['beef_vdw'] = ['beef_vdw_E0', 'beef_vdw_zpe']
        self.c.rankings_rev['custom'] = 9
        self.c.rankings[9] = ['custom']

        self.c.energies['custom1'] = [nan]*len(self.c.energies.index.values)
        self.c.energies['custom2'] = [nan]*len(self.c.energies.index.values)
        self.c.energies.loc[CanonSmiles('OC[Pt]'), new_keys] = [10., 6.]
        self.c.energies.loc[CanonSmiles('C.[Pt]'), new_keys] = [11., 5.]
        self.c.energies.loc[CanonSmiles('CO.[Pt]'), new_keys] = [13., 4.5]
        self.c.energies.loc[CanonSmiles('C[Pt]'), new_keys] = [8., 3.2]
        # reset since previous function sets it to nan
        self.c.energies.loc[CanonSmiles('CO.[Pt]'), 'beef_vdw_zpe'] = -910.330881

        def custom_hrxn_fcn(del_nrg:dict, *args:str) -> float:
            """
            Similar to normal Hrxn composition function found in autocbh.hrxnHelpers
            but divides answer by 500

            :del_nrg:   [dict] Contains energy (Hartree) values.
            :**args:    [str] Dictionary keys to sum for single point energy.
                            Typically the single point energy and zpe.
            """
            hartree_to_kJpermole = 2625.499748 / 500 # (kJ/mol) / Hartree
            hrxn = (sum(del_nrg[k] for k in args)) * hartree_to_kJpermole
            return hrxn

        hrxn_true = stoichiometric_math(energies_df=self.c.energies, full_rxn=self.cbh1_rxn,
                                        species=self.species,
                                        columns=['beef_vdw_E0', 'beef_vdw_zpe'],
                                        convert_from_hartree_to_kjmol=True)
        hrxn_true_custom = 1/500*stoichiometric_math(energies_df=self.c.energies,
                                                     full_rxn=self.cbh1_rxn, species=self.species,
                                                     columns=['custom1', 'custom2'],
                                                     convert_from_hartree_to_kjmol=True)
        hf_partial_calc = stoichiometric_math(energies_df=self.c.energies, full_rxn=self.cbh1_rxn,
                                              species=self.species, columns='DfH',
                                              convert_from_hartree_to_kjmol=False)
        hf_true = [np.round(v - hf_partial_calc, 8)
                   for v in [hrxn_true, hf_partial_calc]]
        hf_true_custom = [np.round(v - hf_partial_calc, 8)
                          for v in [hrxn_true_custom, hf_partial_calc]]

        hrxn, hf = self.c.Hf(s=CanonSmiles('OC[Pt]'), rxn=self.cbh1_rxn,
                             skip_precursor_check=True, hrxn_fcns={'custom':custom_hrxn_fcn})
        hrxn = {k:np.round(v,8) for k, v in hrxn.items()}
        hf = {k:np.round(v,8) for k, v in hf.items()}
        hrxn_correct_dict = {'beef_vdw':np.round(hrxn_true,8), 'ref':np.round(hf_partial_calc,8),
                             'custom':np.round(hrxn_true_custom, 8)}
        hf_correct_dict = {'beef_vdw':hf_true[0], 'ref':hf_true[1], 'custom': hf_true_custom[0]}
        assert hrxn == hrxn_correct_dict and hf == hf_correct_dict

class TestWeightingSchemeHf:

    ranking_file = 'tests/dummy_data/rankings.yaml'
    methods_keys_file = 'tests/dummy_data/methods_keys.yaml'
    c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=ranking_file,
                method_keys_path=methods_keys_file)
    # OC[Pt]
    s = CanonSmiles('OC[Pt]')
    cbh1_rxn_full = convert_dict_keys_to_CanonSmiles({'OC[Pt]':-1, 'C.[Pt]':-1, 'CO.[Pt]':1, 'C[Pt]':1})
    cbh1_rxn = convert_dict_keys_to_CanonSmiles({'C.[Pt]':-1, 'CO.[Pt]':1, 'C[Pt]':1})

    new_rxn_1_full = convert_dict_keys_to_CanonSmiles({'OC[Pt]':-1, 'C.[Pt]':-1, 'CO.[Pt]':1})
    new_rxn_2_full = convert_dict_keys_to_CanonSmiles({'OC[Pt]':-1, 'C.[Pt]':-1, 'C[Pt]':1})
    new_rxn_1 = convert_dict_keys_to_CanonSmiles({'C.[Pt]':-1, 'CO.[Pt]':1})
    new_rxn_2 = convert_dict_keys_to_CanonSmiles({'C.[Pt]':-1, 'C[Pt]':1})

    keys = ['beef_vdw_E0', 'beef_vdw_zpe', 'DfH', 'DrxnH', 'source']
    methods_keys = ['beef_vdw_E0', 'beef_vdw_zpe']
    c.methods_keys = methods_keys
    c.methods_keys_dict = {'beef_vdw': methods_keys}
    c.energies.loc[CanonSmiles('OC[Pt]'), keys] = [-909.715635, 0.041159, 0., 0., nan]
    c.energies.loc[CanonSmiles('C.[Pt]'), keys] = [-889.101433, 0.044577, -81.3, 0., 'exp']
    c.energies.loc[CanonSmiles('CO.[Pt]'), keys] = [-910.330881, 0.052809, -245.0, 0., 'exp']
    c.energies.loc[CanonSmiles('C[Pt]'), keys] = [-888.477198, 0.036051, -47.2, 0., 'exp']

    def test_single_rxn_weight(self):
        self.c.rxns[self.s] = {'H':self.cbh1_rxn}
        labels = ['H']
        weighted_hrxn, weighted_hf = self.c._weighting_scheme_Hf(s=self.s, labels=labels, skip_precursor_check=True, hrxn_fcns={})

        hrxn_true = stoichiometric_math(energies_df=self.c.energies, full_rxn=self.cbh1_rxn_full, 
                                        species=list(self.cbh1_rxn_full.keys()), columns=self.methods_keys, 
                                        convert_from_hartree_to_kjmol=True)
        hf_partial_calc = stoichiometric_math(energies_df=self.c.energies, full_rxn=self.cbh1_rxn_full, 
                                              species=list(self.cbh1_rxn_full.keys()), columns='DfH', 
                                              convert_from_hartree_to_kjmol=False)
        hf_true = [np.round(v - hf_partial_calc, 8) for v in [hrxn_true, hf_partial_calc]]

        weighted_hrxn = {k:np.round(v, 8) for k, v in weighted_hrxn.items()}
        weighted_hf = {k:np.round(v, 8) for k, v in weighted_hf.items()}
        assert weighted_hrxn == {'beef_vdw':abs(np.round(hrxn_true,8)), 'ref':abs(np.round(hf_partial_calc,8))} and weighted_hf == {'beef_vdw':hf_true[0], 'ref':hf_true[1]}

    def test_multiple_rxn_weight(self):
        full_rxns = {
            '1': self.new_rxn_1_full,
            '2': self.new_rxn_2_full
        }
        species = {k: list(v.keys()) for k, v in full_rxns.items()}
        self.c.rxns[self.s] = {
            '1': self.new_rxn_1,
            '2': self.new_rxn_2
        }
        labels = ['1', '2']
        weighted_hrxn, weighted_hf = self.c._weighting_scheme_Hf(s=self.s, labels=labels,
                                                                 skip_precursor_check=True,
                                                                 hrxn_fcns={})

        hrxns = []
        hf_partial_calcs = []
        for l in labels:
            hrxns.append(stoichiometric_math(energies_df=self.c.energies, full_rxn=full_rxns[l], species=species[l], 
                                       columns=self.methods_keys, convert_from_hartree_to_kjmol=True))
            hf_partial_calcs.append(stoichiometric_math(energies_df=self.c.energies, full_rxn=full_rxns[l], species=species[l],
                                                        columns='DfH', convert_from_hartree_to_kjmol=False))
        denom = abs(1/hrxns[0]) + abs(1/hrxns[1])
        weight1 = abs(1/hrxns[0]) / denom
        weight2 = abs(1/hrxns[1]) / denom

        denom2 = abs(1/hf_partial_calcs[0]) + abs(1/hf_partial_calcs[1])
        weight_ref1 = abs((1/hf_partial_calcs[0])) / denom2
        weight_ref2 = abs((1/hf_partial_calcs[1])) / denom2

        hf_true1_beef = weight1*(hrxns[0] - hf_partial_calcs[0])
        hf_true2_beef = weight2*(hrxns[1] - hf_partial_calcs[1])

        weighted_hrxn_true = {'beef_vdw': np.round(np.abs(hrxns[0])*weight1 + np.abs(hrxns[1])*weight2, 8),
                          'ref': np.round(np.abs(hf_partial_calcs[0])*weight_ref1 + np.abs(hf_partial_calcs[1])*weight_ref2, 8)}
        weighted_hrxn = {k:np.round(v, 8) for k, v in weighted_hrxn.items()}

        weighted_hf_true = {'beef_vdw':np.round(hf_true1_beef + hf_true2_beef, 8),
                            'ref': 0.0}
        weighted_hf = {k:np.round(v, 8) for k, v in weighted_hf.items()}
        assert weighted_hrxn_true == weighted_hrxn and weighted_hf == weighted_hf_true

    def test_single_rxn_weight_None(self):
        self.c.rxns[self.s] = {'H':None}
        weighted_hrxn, weighted_hf = self.c._weighting_scheme_Hf(s=self.s, labels=['H'], skip_precursor_check=True, hrxn_fcns={})
        assert np.isnan(tuple(weighted_hf.values())[0]) and np.isnan(tuple(weighted_hrxn.values())[0])

    def test_single_rxn_weight_empty_dict(self):
        self.c.rxns[self.s] = {'H':{}}
        weighted_hrxn, weighted_hf = self.c._weighting_scheme_Hf(s=self.s, labels=['H'], skip_precursor_check=True, hrxn_fcns={})
        assert np.isnan(tuple(weighted_hf.values())[0]) and np.isnan(tuple(weighted_hrxn.values())[0])

    def test_mult_rxn_weight_both_None_or_empty_dict(self):
        self.c.rxns[self.s] = {'H':{}, 'F':None}
        weighted_hrxn, weighted_hf = self.c._weighting_scheme_Hf(s=self.s, labels=['H', 'F'], skip_precursor_check=True, hrxn_fcns={})
        assert np.isnan(tuple(weighted_hf.values())[0]) and np.isnan(tuple(weighted_hrxn.values())[0])

    def test_mult_rxn_weight_hetero_None_empty_real_value(self):
        full_rxns = {
            '1': self.new_rxn_1_full,
            '2': self.new_rxn_2_full
        }
        species = {k: list(v.keys()) for k, v in full_rxns.items()}
        self.c.rxns[self.s] = {
            '1': self.new_rxn_1,
            '2': self.new_rxn_2,
            'H': None,
            'F': {}
        }
        labels = ['1', '2', 'H', 'F']
        weighted_hrxn, weighted_hf = self.c._weighting_scheme_Hf(s=self.s, labels=labels, skip_precursor_check=True, hrxn_fcns={})

        hrxns = []
        hf_partial_calcs = []
        for l in labels[:2]:
            hrxns.append(stoichiometric_math(energies_df=self.c.energies, full_rxn=full_rxns[l],
                                             species=species[l], columns=self.methods_keys,
                                             convert_from_hartree_to_kjmol=True))
            hf_partial_calcs.append(stoichiometric_math(energies_df=self.c.energies,
                                                        full_rxn=full_rxns[l], species=species[l],
                                                        columns='DfH',
                                                        convert_from_hartree_to_kjmol=False))
        denom = abs(1/hrxns[0]) + abs(1/hrxns[1])
        weight1 = abs(1/hrxns[0]) / denom
        weight2 = abs(1/hrxns[1]) / denom

        denom2 = abs(1/hf_partial_calcs[0]) + abs(1/hf_partial_calcs[1])
        weight_ref1 = abs((1/hf_partial_calcs[0])) / denom2
        weight_ref2 = abs((1/hf_partial_calcs[1])) / denom2

        hf_true1_beef = weight1*(hrxns[0] - hf_partial_calcs[0])
        hf_true2_beef = weight2*(hrxns[1] - hf_partial_calcs[1])

        weighted_hrxn_true = {'beef_vdw': np.round(np.abs(hrxns[0])*weight1 + np.abs(hrxns[1])*weight2, 8),
                          'ref': np.round(np.abs(hf_partial_calcs[0])*weight_ref1 + np.abs(hf_partial_calcs[1])*weight_ref2, 8)}
        weighted_hrxn = {k:np.round(v, 8) for k, v in weighted_hrxn.items()}

        weighted_hf_true = {'beef_vdw':np.round(hf_true1_beef + hf_true2_beef, 8),
                            'ref': 0.0}
        weighted_hf = {k:np.round(v, 8) for k, v in weighted_hf.items()}
        assert weighted_hrxn_true == weighted_hrxn and weighted_hf == weighted_hf_true


class TestChooseBestMethod:

    ranking_file = 'tests/dummy_data/rankings.yaml'
    methods_keys_file = 'tests/dummy_data/methods_keys.yaml'
    c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=ranking_file,
                method_keys_path=methods_keys_file)

    def test_Hrxn_all_nan(self):
        hrxn = {'method1':nan, 'method2': nan, 'method3':nan}
        hf_dummy = {'method1': 1., 'method2': 2., 'method3':3.}
        label = 'test'

        tuple_of_nans = self.c._choose_best_method(hrxn=hrxn, hf=hf_dummy, label=label)
        assert all([np.isnan(val) for val in tuple_of_nans])

    def test_rank_0_and_skip_rank_and_rank_with_1(self):
        """ It should ignore rank 0 and rank 1 since no data. """
        self.c.rankings[0] = ['zero']
        self.c.rankings_rev['zero'] = 0

        hrxn = {'zero': 4., 'beef_vdw': 6.}
        hf = {'zero': 7., 'beef_vdw': 9.}
        label = 'test'

        tuple_ans = self.c._choose_best_method(hrxn=hrxn, hf=hf, label=label)
        assert tuple_ans == (6., 9., 'test//beef_vdw')

    def test_skip_rank_with_multiple_nans(self):
        hrxn = {'zero': 4., 'ATcT':nan, 'exp':nan, 'beef_vdw': 6.}
        hf = {'zero': 7., 'ATcT':nan, 'exp':nan, 'beef_vdw': 9.}
        label = 'test'

        tuple_ans = self.c._choose_best_method(hrxn=hrxn, hf=hf, label=label)
        assert tuple_ans == (6., 9., 'test//beef_vdw')

    def test_rank_with_multiple_vals_but_one_not_nan(self):
        hrxn = {'zero': 4., 'ATcT':nan, 'exp':1., 'beef_vdw': 6.}
        hf = {'zero': 7., 'ATcT':nan, 'exp':2., 'beef_vdw': 9.}
        label = 'test'

        tuple_ans = self.c._choose_best_method(hrxn=hrxn, hf=hf, label=label)
        assert tuple_ans == (1., 2., 'test//exp')

    def test_multiple_vals_with_same_rank(self):
        hrxn = {'zero': 4., 'ATcT':1., 'exp':1., 'beef_vdw': 6.}
        hf = {'zero': 7., 'ATcT':2., 'exp':2., 'beef_vdw': 9.}
        label = 'test'

        true_hrxn, true_hf, true_label = self.c._choose_best_method(hrxn=hrxn, hf=hf, label=label)
        assert true_hrxn == 1. and true_hf == 2. and (true_label in ('test//ATcT+exp','test//exp+ATcT'))

class TestDecomposeRxn:
    # TODO: need to do tests for larger datasets
    ranking_file = 'tests/dummy_data/rankings.yaml'
    methods_keys_file = 'tests/dummy_data/methods_keys.yaml'
    c = calcCBH(dataframe_path='tests/dummy_data/adsorbate_data.pkl',
                rankings_path=ranking_file,
                method_keys_path=methods_keys_file)

    smiles = 'COCO.[Pt]'
    cbh = buildCBH(smiles=smiles, saturate=1, allow_overshoot=False,
                   ignore_F2=True, surface_smiles='[Pt]')
    c.rxns[smiles] = {}
    c.error_messages[smiles] = []

    def test_partial_nan_energies_in_precursor_df_rct(self):
        delete_smiles = 'O.[Pt]'
        self.c.energies.loc[delete_smiles, ['beef_vdw_E0']] = nan

        rung = self.c._decompose_rxn(s=self.smiles, test_rung=1,
                                     cbh_rcts=self.cbh.cbh_rcts,
                                     cbh_pdts=self.cbh.cbh_pdts, label='H')
        check_rxn = add_dicts(self.cbh.cbh_pdts[1],
                              {k : -v for k, v in self.cbh.cbh_rcts[1].items()})
        assert rung == '1' and self.c.rxns[self.smiles]['H'] == check_rxn

    def test_all_nan_energies_in_precursor_df_rct(self):
        delete_smiles = 'O.[Pt]'
        self.c.energies.loc[delete_smiles, ['beef_vdw_E0', 'beef_vdw_zpe']] = nan

        rung = self.c._decompose_rxn(s=self.smiles, test_rung=1,
                                     cbh_rcts=self.cbh.cbh_rcts,
                                     cbh_pdts=self.cbh.cbh_pdts, label='H')
        check_rxn = add_dicts(self.cbh.cbh_pdts[1],
                              {k : -v for k, v in self.cbh.cbh_rcts[1].items()})
        assert rung == '1' and self.c.rxns[self.smiles]['H'] == check_rxn

    def test_missing_species_in_dataframe(self):
        delete_smiles = 'O.[Pt]'
        self.c.energies.drop(delete_smiles, inplace=True)

        rung = self.c._decompose_rxn(s=self.smiles, test_rung=1,
                                     cbh_rcts=self.cbh.cbh_rcts,
                                     cbh_pdts=self.cbh.cbh_pdts, label='H')
        assert rung == '1' and self.c.rxns[self.smiles]['H'] == {}


class TestCheckRungEquivalency:
    pass

class TestCheckAltRungUsability:
    pass
