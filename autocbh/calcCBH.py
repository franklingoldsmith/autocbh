import CBH
from CBH import add_dicts
import numpy as np
import pandas as pd
from numpy import nan, isnan
from rdkit.Chem import MolFromSmiles, AddHs, CanonSmiles, GetPeriodicTable
from data.molData import load_rankings, generate_database, generate_alternative_rxn_file
from hrxnHelpers import anl0_hrxn, sum_Hrxn
import os
import yaml
from itertools import compress
from copy import copy
import sys
sys.path.append('.')


class calcCBH:
    """
    This class handles the hierarchical calculation of heats of formation
    using CBH schemes.

    ATTRIBUTES
    ----------
    :methods:           [list(str)] User inputted list of method names to 
                            use for calculation of HoF. If empty, use all 
                            available methods.
    :methods_keys:      [list(str)] All keys of energy properties needed to 
                            compute the Hrxn/Hf.
    :methods_keys_dict: [dict] Each method's keys needed to compute energy 
                            properties. {method : list(keys)}
    :rankings:          [dict] Dictionary of the rankings for different 
                            levels of theory. {rank # : list(methods)}
    :error_messages:    [dict] Dictionary holding logged error messages.
                            Use calcCBH.print_errors() method.

    METHODS
    -------
    generate_CBHs   :   Generate a list of DataFrames of coefficients for 
                            CBH schemes
    calc_Hf         :   Stores the best heats of formation in a given database
                            in a hiearchical manner using CBH rungs
    Hf              :   Workhorse calculates the heats of formation and
                            reaction of a given species
    calc_Hf_allrungs:   Calculate the heats of formation and reaction 
                            at each rung for a given species
    """

    def __init__(self, methods:list=[], 
                 dataframe_path:str=None, 
                 method_keys_path:str='../data/methods_keys.yaml', 
                 rankings_path:str='../data/rankings.yaml', 
                 alternative_rxn_path:str=None, 
                 force_generate_database:str=None, 
                 force_generate_alternative_rxn:str=None, 
                 zero_out_heats:bool=False):
        """
        ARGUMENTS
        ---------
        :methods:                           [list] (default=[])
                List of method names to use for calculation of HoF. If empty, 
                use all available methods.
                            
        :dataframe_path:                    [str] (default=None)
                Path to a PICKLE file of a dataframe to be used for self.energies.
                The index contains RDKit canonized SMILES (rdkit.Chem.CanonSmiles)
                for each species. The columns contain each QM energy corresponding to
                a method. Finally the columns "DfH", "DrxnH", and "source" must be included.
                "DfH" is the heat of formation, "DrxnH" is the heat of reaction, "source"
                is where DfH was derived. Species with unknown "DfH" should ensure that
                "DfH" and "DrxnH" are 0 and "source" is np.nan. This final step can be done
                with the zero_out_heats argument.

        :method_keys_path:                  [str] (default='../data/methods_keys.yaml')
                Path to a YAML file that maps the column names of the self.energies 
                dataframe to a method (ex. method_1: [method_1_E, method_1_zpe])
        
        :rankings_path:                     [str] (default='../data/rankings.yaml')
                Path to a YAML file that maps each method to a ranking.

        :alternative_rxn_path:              [str] (default=None)
                Path to a YAML file that contains the different alternative reactions
                for a given species.

        :force_generate_database:           [str] (default=None)
                Path to a folder containing individual species data in YAML files
                to generate the self.energies dataframe.
                Typically held at: 'data/molecule_data'

        :force_generate_alternative_rxn:    [str] (default=None) 
                Path to a folder containing individual species data in YAML files to
                generate and use an alternative_rxn.yaml file.
                
        :zero_out_heats:                    [bool] (default=False)
                Will set all heats of reaction to 0 and all heats of formation for
                sources that are not rank 1 (experimental) to 0.
        """

        if force_generate_database is None and dataframe_path is None:
            raise ValueError('Either a dataframe pkl filepath or a path to a folder holding molecule data in YAML files must be provided to the "dataframe_path" or "force_generate_database" arguments.')

        # Generate Database
        if force_generate_database:
            self.energies = pd.DataFrame(generate_database(force_generate_database, ranking_path=rankings_path)[0])
        else:
            if dataframe_path:
                self.energies = pd.read_pickle(dataframe_path)
            
            # This will not happen
            else:
                self.energies = pd.read_pickle('./data/pfas_energies.pkl')
                # self.energies = generate_database('data/molecule_data/')[0] # something is different
                self.energies[['DrxnH']] = 0 # add delta heat of reaction column --> assume 0 for ATcT values
                ###
                # This stuff should go into test cases
                # self.energies.drop('CC(F)(F)F',axis=0, inplace=True) # for testing
                # self.energies.loc['CC(C)(F)F', ['avqz','av5z','zpe','ci_DK','ci_NREL','core_0_tz','core_X_tz','core_0_qz','core_X_qz',
                # 'ccT','ccQ','zpe_harm','zpe_anharm','b2plypd3_zpe','b2plypd3_E0','f12a','f12b','m062x_zpe',
                # 'm062x_E0','m062x_dlpno','wb97xd_zpe','wb97xd_E0','wb97xd_dlpno']] = nan

        # sort by num C then by SMILES length
        max_C = max([i.count('C') for i in self.energies.index])
        self.energies.sort_index(key=lambda x: x.str.count('C')*max_C+x.str.len(),inplace=True)

        # Load the methods to use
        with open(method_keys_path, 'r') as f:
            self.methods_keys_dict = yaml.safe_load(f)

        # choose which methods to use
        if len(methods)==0:
            # use all available methods in methods_keys dictionary
            self.methods = []
            self.methods_keys = []
            methods_to_remove = [] # methods in the method_keys_path that don't exist in the provided energies dataframe
            for m in self.methods_keys_dict:
                if all([_m in self.energies.columns for _m in self.methods_keys_dict[m]]):
                    self.methods_keys.extend(self.methods_keys_dict[m])
                    self.methods_keys = list(set(self.methods_keys))
                else:
                    methods_to_remove.append(m)
            if len(self.methods_keys) == 0:
                raise KeyError(f'None of the method keys found in "{method_keys_path}" were found in the provided DataFrame columns. \nPlease ensure that correct keys are in either the YAML file or the DataFrame.')
            for m in methods_to_remove:
                del self.methods_keys_dict[m]
        else:
            methods_keys_list = list(self.methods_keys_dict.keys())
            for m in methods_keys_list:
                if m not in methods:
                    del self.methods_keys_dict[m]
            self.methods = methods
            self.methods_keys = []
            for m in self.methods:
                self.methods_keys.extend(self.methods_keys_dict[m])
                self.methods_keys = list(set(self.methods_keys))

        if force_generate_database:
            self.energies = self.energies[self.methods_keys+['source', 'DfH', 'DrxnH']]
        
        # Load rankings
        # maps rank to methods
        self.rankings = load_rankings(rankings_path)
        # remove items from rankings list that aren't needed based on user selected methods
        del_methods = {} # stores methods to delete from self.rankings
        for rank in self.rankings.keys():
            if rank == 0 or rank == 1:
                continue
            for m in self.rankings[rank]:
                if m not in self.methods_keys_dict.keys():
                    if rank not in del_methods.keys():
                        del_methods[rank] = [m]
                    else:
                        del_methods[rank].append(m)
        for rank, method_list in del_methods.items():
            if len(del_methods[rank]) == len(self.rankings[rank]):
                del self.rankings[rank]
            else:
                self.rankings[rank].remove(*method_list)
        
        # maps each method to rank
        self.rankings_rev = {}
        for k,l in self.rankings.items():
            for v in l:
                self.rankings_rev[v] = k

        if force_generate_alternative_rxn is not None and os.path.isdir(force_generate_alternative_rxn) and alternative_rxn_path is None:
            generate_alternative_rxn_file(force_generate_alternative_rxn, 'alternative_rxn')
            alternative_rxn_path = os.path.join(force_generate_alternative_rxn, 'alternative_rxn.yaml')

        self.alternative_rxn = {}
        if alternative_rxn_path is not None:
            with open(alternative_rxn_path, 'r') as f:
                self.alternative_rxn = yaml.safe_load(f)

        if zero_out_heats:
            # zero out heats of formation for species without experimental heats of formation
            # zero out heats of reaction for all species
            non_exp_species = [s for s in self.energies.index.values if (type(self.energies.loc[s, 'source']) is not str and isnan(self.energies.loc[s, 'source'])) or self.rankings_rev[self.energies.loc[s, 'source']] != 1]
            self.energies.loc[non_exp_species, ['DfH']] = 0.0
            self.energies.loc[:, ['DrxnH']] = 0.0

        self.error_messages = {}


    def calc_Hf(self, saturate:list=[1], priority:str="abs_coeff", max_rung:int=None, alt_rxn_option:str=None, surface_smiles=None):
        """
        Calculate the heats of formation of species that do not have reference 
        values using the highest possible CBH scheme with the best possible level 
        of theory. This is done for the entire database.
        
        ARGUMENTS
        ---------
        :self:
        :saturate:      [list(int) or list(str)] (default=[1] for hydrogen)
                List of integers representing the atomic numbers of the elements to 
                saturate precursor species. String representations of the elements 
                also works.

        :priority:      [str] (default="abs_coeff")
                    Rung selection criteria.
            - "abs_coeff" 
                The reaction with the minimum number of total precursors will be 
                prioritized. If there are multiple reactions with the same number, 
                they will be averaged.
                
            - "rel_coeff"
                The reaction with the smallest sum of reaction coefficients will be 
                prioritized. Reactants have negative contribution and products have 
                positive contribution.
                
                Note: In most/all cases, all reactions sum to 0. This means that 
                the highest possible rungs for all saturation schemes and alternative
                reactions will be averaged.

            - "rung"
                The reaction will the prioritized by the highest rung number of a given
                CBH scheme. If there are multiple reactions with the same number,
                they will be averaged.
                
            e.g.) CBH-1-H: CH3CF2CH3 + 3 CH4 --> 2 C2H6 + 2 CH3F
            "abs_coeff":    val = |1 + 3 + 2 + 2| = 7
            "rel_coeff":    val = 2 + 2 - (1 + 3) = 0
            "rung":         val = 1


        :max_rung:      [int] (default=None) Max CBH rung allowed for Hf calculation

        :alt_rxn_option: [str] (default=None)
                    - "ignore" (ignr)
                        Ignore alternative reactions that may exist and only use CBH 
                        priority system. (same result as None)
                    
                    - "best_alt" (best)
                        Always uses the best user-defined alternative reaction for a 
                        species even if better ranking CBH rungs exist.

                    - "avg_alt" (avg)
                        Average all alternative reactions of a species and does not
                        use any default CBH rungs.

                    - "include" (priority)
                        Included alternative reactions and will follow CBH priority 
                        protocol. 
        
        :surface_smiles:    [str] (default=None)
                Valid SMILES string representing the surface atom that the given 
                molecule is adsorbed to or physiosorbed to. Must be a single atom.
                i.e., '[Pt]'

        RETURN
        ------
        :self.energies:  [pd.DataFrame] columns for heat of formation, heat of 
                            reaction, and source.
        """
        
        alt_rxn_option_list = ['ignore', 'best_alt','avg_alt', 'include']
        if alt_rxn_option:
            if type(alt_rxn_option)!= str:
                raise TypeError('Arg "alt_rxn_option" must either be NoneType or str. If str, the options are: "ignore", "best_alt", "avg_alt", "include".')
            elif alt_rxn_option not in alt_rxn_option_list:
                raise NameError('The available options for arg "alt_rxn_option" are: "ignore", "best_alt", "avg_alt", "include".')
        
        priority_list = ["abs_coeff", "rel_coeff", "rung"]
        if not isinstance(priority, str):
            raise TypeError(f'Arg "priority" must be a str. Instead, {type(priority)} was given. The options are: "abs_coeff", "rel_coeff", "rung".')
        if priority not in priority_list:
            raise NameError(f'The available options for arg "priority" are: "abs_coeff", "rel_coeff", "rung". {priority} was given instead.')

        ptable = GetPeriodicTable()
        saturate_syms = []
        saturate_nums = []
        for sat in saturate:
            if type(sat) == str:
                try:
                    saturate_nums.append(ptable.GetAtomicNumber(sat))
                    saturate_syms.append(sat)
                except:
                    KeyError("Provided str of saturation element is not present in Periodic Table.")
            elif type(sat) == int:
                try:
                    saturate_syms.append(ptable.GetElementSymbol(sat))
                    saturate_nums.append(sat)
                except:
                    KeyError("Provided int of saturation element is not present in Periodic Table.")
            else:
                TypeError("Elements within saturation list must be int or str.")
        
        if surface_smiles:
            try:
                surface_smiles = CanonSmiles(surface_smiles)
            except:
                raise ValueError(f'Arg "surface_smiles" must be a valid SMILES string. Instead, "{surface_smiles}" was given.')
            # make sure surface smiles is just an element
            try:
                ptable.GetAtomicNumber(surface_smiles.replace('[','').replace(']',''))
            except:
                raise ValueError(f'Arg "surface_smiles" must be an element. Instead, "{surface_smiles}" was given.')
            self.surface_smiles = surface_smiles
        else:
            self.surface_smiles=None

        def choose_best_method_and_assign(Hrxn_d:dict, Hf_d:dict, label:str):
            """Chooses best method then assigns calculated values to self.energies dataframe."""
            final_Hrxn, final_Hf, label = self._choose_best_method(Hrxn_d, Hf_d, label)
            self.energies.loc[s, 'DfH'] = final_Hf
            self.energies.loc[s, 'DrxnH'] = final_Hrxn
            self.energies.loc[s, 'source'] = label

        Hf = {}
        Hrxn = {}
        # sort by the max distance between two atoms in a molecule
        def simple_sort(x):
            """Sorting algorithm used for smallest to largest
            Computes the longest path across a molecule.
            The np.inf condition is used to avoid any issues with
            physiosorbed species."""
            arr = np.array(CBH.mol2graph(AddHs(MolFromSmiles(x))).shortest_paths())
            return max(arr[arr < np.inf])
        # sorted list of molecules that don't have any reference values
        sorted_species = sorted(self.energies[self.energies['source'].isna()].index.values, key=simple_sort)

        # cycle through molecules from smallest to largest
        for s in sorted_species:
            self.error_messages[s] = []

            if s in self.alternative_rxn and alt_rxn_option and alt_rxn_option != "ignore":
                ##### avg and avg alternative rxns options #####
                if alt_rxn_option == "best_alt" or alt_rxn_option == "avg_alt":
                    best_alt = max(self.alternative_rxn[s].keys())
                    if alt_rxn_option == "avg_alt":
                        alt_rxn_keys = list(self.alternative_rxn[s].keys())
                    elif alt_rxn_option == "best_alt":
                        alt_rxn_keys = [best_alt]

                    alt_rxns = self._check_alt_rxn_usability(s, alt_rxn_keys)
                    if alt_rxns:
                        Hrxn[s], Hf[s] = self._weighting_scheme_Hf(s, alt_rxns, skip_precursor_check=True)
                        # Choose the best possible method to assign to the energies dataframe
                        if len(alt_rxns) == 1:
                            label = f'CBH-{list(alt_rxns.keys())[0]}-alt'
                        elif len(alt_rxns) > 1:
                            label = 'CBHavg-('
                            for i, k in enumerate(alt_rxns.keys()):
                                if i > 0:
                                    label += f', '
                                label += f'{k}-alt'
                                if i == len(alt_rxns) - 1:
                                    label += ')'
                                
                        choose_best_method_and_assign(Hrxn[s], Hf[s], label)

                        if len(self.error_messages[s]) == 0:
                            del self.error_messages[s]
                        continue
                    
                ##### prioritize by user defined rung numbers #####
                elif alt_rxn_option == "include" and priority == "rung":
                    alt_rxn_keys = list(self.alternative_rxn[s].keys())
                    alt_rxns = self._check_alt_rxn_usability(s, alt_rxn_keys)
                    if alt_rxns:
                        best_alt = max(alt_rxns.keys())
                        alt_rxns = {best_alt : alt_rxns[best_alt]}
                        
                        cbhs = []
                        cbhs_rungs = []
                        for i, sat in enumerate(saturate_nums):
                            cbhs.append(CBH.buildCBH(s, sat, allow_overshoot=True, surface_smiles=self.surface_smiles))
                            if max_rung is not None:
                                rung = max_rung if max_rung <= cbhs[-1].highest_cbh else cbhs[-1].highest_cbh
                            else:
                                rung = cbhs[-1].highest_cbh
                            cbhs_rungs.append(self._check_rung_usability(s, rung, cbhs[-1].cbh_rcts, cbhs[-1].cbh_pdts, saturate_syms[i]))
                        
                        idx = np.where(np.array(cbhs_rungs) == np.array(cbhs_rungs).max())[0].tolist()
                        cbhs = [cbhs[i] for i in idx]
                        cbhs_rungs = [cbhs_rungs[i] for i in idx]
                        s_syms = [saturate_syms[i] for i in idx]

                        # user rung is better than automated rungs
                        if best_alt > cbhs_rungs[0]:
                            Hrxn[s], Hf[s] = self._weighting_scheme_Hf(s, alt_rxns, skip_precursor_check=True)
                            # Choose the best possible method to assign to the energies dataframe
                            label = f'CBH-{best_alt}-alt'
                        
                        # EQUAL user and automated rungs 
                        elif best_alt == cbhs_rungs[0]:
                            new_rxn_d = {}
                            for i, rxn in enumerate(cbhs):
                                new_rxn_d[i] = self._cbh_to_rxn(s, rxn, cbhs_rungs[i])
                            new_rxn_d[i+1] = alt_rxns[best_alt]
                            Hrxn[s], Hf[s] = self._weighting_scheme_Hf(s, new_rxn_d, skip_precursor_check=True)

                            # Choose the best possible method to assign to the energies dataframe
                            label = f'CBHavg-('
                            for i, sym in enumerate(s_syms):
                                if i > 0:
                                    label += ', '
                                label += str(cbhs_rungs[i]) + '-' + sym
                            label += f', {best_alt}-alt)'

                        # user rung is worse than automated rungs
                        elif best_alt < cbhs_rungs[0]:
                            if len(cbhs_rungs) == 1:
                                label = f'CBH-{str(cbhs_rungs[0])}-{s_syms[0]}'
                            elif len(cbhs_rungs) > 1:
                                label = f'CBHavg-('
                                for i, sym in enumerate(s_syms):
                                    if i > 0:
                                        label += ', '
                                    label += str(cbhs_rungs[i]) + '-' + sym
                                    if i == len(s_syms)-1:
                                        label += ')'
                            
                            Hrxn[s], Hf[s] = self._weighting_scheme_Hf(s, cbhs, skip_precursor_check=True, cbh_rungs=cbhs_rungs)
                        
                        choose_best_method_and_assign(Hrxn[s], Hf[s], label)

                        if len(self.error_messages[s]) == 0:
                            del self.error_messages[s]
                        continue

                ##### prioritize by coeff #####
                elif alt_rxn_option == "include" and priority in ("abs_coeff", "rel_coeff"):
                    alt_rxn_keys = list(self.alternative_rxn[s].keys())
                    alt_rxns = self._check_alt_rxn_usability(s, alt_rxn_keys)
                    if alt_rxns:
                        if priority == "abs_coeff":
                            rank_totalcoeff = {rank : sum(map(abs, alt_rxns[rank][s].values())) for rank in alt_rxns}
                        elif priority == "rel_coeff":
                            rank_totalcoeff = {rank : sum(alt_rxns[rank][s].values()) for rank in alt_rxns}
                        mn = min(rank_totalcoeff.values())
                        mn_keys_alt = [k for k, v in rank_totalcoeff.items() if v == mn]

                        cbhs = []
                        cbhs_rungs = []
                        for i, sat in enumerate(saturate_nums):
                            cbhs.append(CBH.buildCBH(s, sat, allow_overshoot=True, surface_smiles=self.surface_smiles))
                            if max_rung is not None:
                                rung = max_rung if max_rung <= cbhs[-1].highest_cbh else cbhs[-1].highest_cbh
                            else:
                                rung = cbhs[-1].highest_cbh
                            cbhs_rungs.append(self._check_rung_usability(s, rung, cbhs[-1].cbh_rcts, cbhs[-1].cbh_pdts, saturate_syms[i]))
                        idx = np.where(np.array(cbhs_rungs) == np.array(cbhs_rungs).max())[0].tolist()
                        cbhs = [cbhs[i] for i in idx]
                        cbhs_rungs = [cbhs_rungs[i] for i in idx]
                        s_syms = [saturate_syms[i] for i in idx]

                        new_rxn_d = {}
                        for i, rxn in enumerate(cbhs):
                            new_rxn_d[i] = self._cbh_to_rxn(s, rxn, cbhs_rungs[i])
                        if priority == "abs_coeff":
                            cbh_totalcoeff = {i : sum(map(abs, new_rxn_d[i][s].values())) for i in new_rxn_d}
                        elif priority == "rel_coeff":
                            cbh_totalcoeff = {i : sum(new_rxn_d[i][s].values()) for i in new_rxn_d}

                        # if alt_rxn has the lowest total coeff, use alt_rxn
                        if mn < min(cbh_totalcoeff.values()):
                            new_rxn_d = {i: alt_rxns[i] for i in mn_keys_alt}
                            if len(mn_keys_alt) == 1:
                                label = 'CBH-' + str(mn_keys_alt[0]) + '-alt'
                            else:
                                label = 'CBHavg-('
                                for i, idx in enumerate(mn_keys_alt):
                                    if i > 0:
                                        label += ', '
                                    label += str(idx)+'-alt'
                                    if i == len(mn_keys_alt) - 1:
                                        label += ')'

                        # if alt_rxn has the same total coeff as cbh, use average of both
                        elif mn == min(cbh_totalcoeff.values()):
                            mn_keys_cbh = [k for k, v in cbh_totalcoeff.items() if v == mn]
                            new_rxn_d = [new_rxn_d[k] for k in mn_keys_cbh]
                            new_rxn_d.extend([alt_rxns[i] for i in mn_keys_alt])
                            new_rxn_d = {i:v for i, v in enumerate(new_rxn_d)}
                            
                            label = f'CBHavg-('
                            for i, idx in enumerate(mn_keys_cbh):
                                if i > 0:
                                    label += ', '
                                label += f'{cbhs_rungs[idx]}-{s_syms[idx]}'
                            for i, idx in enumerate(mn_keys_alt):
                                label += f', {idx}-alt'
                                if i == len(mn_keys_alt) - 1:
                                    label += ')'

                        # if alt_rxn has higher total coeff, use cbh protocol
                        elif mn > min(cbh_totalcoeff.values()):
                            mn = min(cbh_totalcoeff.values())
                            idx = [i for i, v in cbh_totalcoeff.items() if v == mn]
                            cbhs = [cbhs[i] for i in idx]
                            cbhs_rungs = [cbhs_rungs[i] for i in idx]
                            s_syms = [saturate_syms[i] for i in idx]
                            
                            if len(cbhs_rungs) == 1:
                                label = f'CBH-{str(cbhs_rungs[0])}-{s_syms[0]}'
                            elif len(cbhs_rungs) > 1:
                                label = f'CBHavg-('
                                for i, sym in enumerate(s_syms):
                                    if i > 0:
                                        label += ', '
                                    label += str(cbhs_rungs[i]) + '-' + sym
                                    if i == len(s_syms)-1:
                                        label += ')'

                        Hrxn[s], Hf[s] = self._weighting_scheme_Hf(s, new_rxn_d, skip_precursor_check=True)
                        choose_best_method_and_assign(Hrxn[s], Hf[s], label)
                        if len(self.error_messages[s]) == 0:
                            del self.error_messages[s]
                        continue

            cbhs = []
            cbhs_rungs = []
            for i, sat in enumerate(saturate_nums):
                cbhs.append(CBH.buildCBH(s, sat, allow_overshoot=True, surface_smiles=self.surface_smiles))
                if max_rung is not None:
                    rung = max_rung if max_rung <= cbhs[-1].highest_cbh else cbhs[-1].highest_cbh
                else:
                    rung = cbhs[-1].highest_cbh
                cbhs_rungs.append(self._check_rung_usability(s, rung, cbhs[-1].cbh_rcts, cbhs[-1].cbh_pdts, saturate_syms[i]))

            # prioritize reactions by total coefficients or by rung number
            if priority in ("abs_coeff", "rel_coeff"):
                if priority == "rel_coeff":
                    cbh_totalcoeff = {i : -1 + sum(cbh.cbh_pdts[cbhs_rungs[i]].values()) - sum(cbh.cbh_rcts[cbhs_rungs[i]].values()) for i, cbh in enumerate(cbhs)}
                elif priority == "abs_coeff":
                    cbh_totalcoeff = {i : 1 + sum(cbh.cbh_pdts[cbhs_rungs[i]].values()) + sum(cbh.cbh_rcts[cbhs_rungs[i]].values()) for i, cbh in enumerate(cbhs)}
                mn = min(cbh_totalcoeff.values())
                idx = [i for i, v in cbh_totalcoeff.items() if v == mn]
            else:
                idx = np.where(np.array(cbhs_rungs) == np.array(cbhs_rungs).max())[0].tolist()

            cbhs = [cbhs[i] for i in idx]
            cbhs_rungs = [cbhs_rungs[i] for i in idx]
            s_syms = [saturate_syms[i] for i in idx]
            
            if len(cbhs_rungs) == 1:
                label = f'CBH-{str(cbhs_rungs[0])}-{s_syms[0]}'
            elif len(cbhs_rungs) > 1:
                label = f'CBHavg-('
                for i, sym in enumerate(s_syms):
                    if i > 0:
                        label += ', '
                    label += str(cbhs_rungs[i]) + '-' + sym
                    if i == len(s_syms)-1:
                        label += ')'

            Hrxn[s], Hf[s] = self._weighting_scheme_Hf(s, cbhs, skip_precursor_check=True, cbh_rungs=cbhs_rungs)
            
            if len(self.error_messages[s]) == 0:
                del self.error_messages[s]

            # Choose the best possible method to assign to the energies dataframe
            choose_best_method_and_assign(Hrxn[s], Hf[s], label)

        if len(self.error_messages.keys()) != 0:
            print(f'Process completed with errors in {len(self.error_messages.keys())} species')
            print(f'These errors are likely to have propagated for the calculation of heats of formation of larger species.')
            print(f'Updating the reference values of species in the database will improve accuracy of heats of formation.')
            print(f'To inspect the errors, run the calcCBH.print_errors() method.')

        return self.energies[['DfH', 'DrxnH', 'source']]


    def _cbh_to_rxn(self, s: str, cbh: CBH.buildCBH, cbh_rung: int) -> dict:
        """
        Helper method to generate a dictionary from a cbh's reactants
        and products. This negates the reactants and adds the species
        itself as a reactant. It uses the CBH.buildCBH object.

        ARGUMENTS
        ---------
        :s:         [str] SMILES string of the target species.
        :cbh:       [CBH.buildCBH obj] CBH scheme of the target species.
        :cbh_rung:  [int] CBH rung number to calculate the HoF

        RETURNS
        -------
        :rxn:       [dict] Includes the negated reactant coefficient
                        and product coefficients. The species, s, is in
                        the dictionary with a negated coefficient.
                        {species SMILES: {reactant SMILES: coefficient}}
        """
        
        rxn = {} 
        # add rxn of a target molecule's highest possible CBH level
        rxn[s] = cbh.cbh_pdts[cbh_rung] # products
        rxn[s].update((p,coeff*-1) for p,coeff in cbh.cbh_rcts[cbh_rung].items()) # negative reactants
        rxn[s].update({s:-1}) # target species

        return rxn


    def Hf(self, s: str, rxn:dict, skip_precursor_check=False) -> tuple:
        """
        Calculate the heat of formation (and reaction) given a species 
        and it's reaction dictionary. The reactants must be negated.

        ARGUMENTS
        ---------
        :s:         [str] 
            SMILES string of the target species.
        
        :rxn:       [dict] 
            Includes the negated reactant coefficient and product
            coefficients. The species, s, is in the dictionary
            with a negated coefficient.

        :skip_precursor_check:  [bool] (default=False)
            Whether skip the procedure to check if precursors exist in 
            database or if they have real values. Suggested to skip 
            only when the cbh_rung is validated after 
            check_rung_usability method is used.

        RETURNS
        -------
        (Hrxn, Hf)  [tuple]
            :Hrxn:  [dict] 
                Heat of reaction calculated for different levels of 
                theory using CBH.
                {'ref' : val, *method : val}

            :Hf:    [dict] 
                Heat of formation calculated for each of the same levels
                of theory in Hrxn. 
                {'ref' : val, *method : val}
        """

        if not skip_precursor_check:
            # check if there are any precursors that don't exist in the database
            precursors_not_tabulated = [p for p in rxn[s] if p not in self.energies.index.values]
            if len(precursors_not_tabulated) != 0:
                print(f'Missing the following species in the database for species: {s}:')
                for pnt in precursors_not_tabulated:
                    print(f'\t{pnt}')
                return nan, nan
            
            # TODO: Fix this since it will break the caclulation and user can't control order
            for precursor in rxn[s]:
                if self.energies.loc[precursor,'source'].isna():
                    print(f'Must restart with this molecule first: {precursor}')
                    return nan, nan
        
        # energy columns needed for calculation
        nrg_cols = ['DfH']
        nrg_cols.extend(self.methods_keys)

        # heat of rxn
        rxn_key_order = list(rxn[s].keys())
        coeff_vec = np.array(list(rxn[s].values())) * -1 # products are negative, and reactants are positive
        nrg_arr = self.energies.loc[rxn_key_order, nrg_cols].values
        
        matrix_mult = np.matmul(coeff_vec, nrg_arr)

        delE = {nrg_cols[i]:matrix_mult[i] for i in range(len(nrg_cols))}
        # Subtract off DfH from the energies column in case it is not 0
        # we must assert that the DfH is not computed for the target species
        delE['DfH'] = delE['DfH'] - self.energies.loc[s, 'DfH']
        
        Hrxn = {'ref':delE['DfH']}
        Hrxn = {**Hrxn, **dict(zip(self.methods, np.full(len(self.methods), nan)))}

        # in most cases, it is just summing the values in method_keys for a given method
        for rank, m_list in self.rankings.items():
            # This is assuming rank=1 is experimental and already measures Hf
            if rank == 0 or rank == 1:
                continue
            for m in m_list:
                if m == 'anl0' and True not in self.energies.loc[rxn[s].keys(), ['avqz', 'av5z', 'zpe']].isna():
                    Hrxn['anl0'] = anl0_hrxn(delE)
                elif 0 not in self.energies.loc[rxn[s].keys(), self.methods_keys_dict[m]].values:
                    Hrxn[m] = sum_Hrxn(delE, *self.methods_keys_dict[m])
        
        Hf = {k:v - Hrxn['ref'] for k,v in Hrxn.items()}

        return Hrxn, Hf


    def _weighting_scheme_Hf(self, s, cbh_or_altrxn:list or dict, skip_precursor_check=False, cbh_rungs:list=None) -> dict:
        """
        Weighting scheme the Hf values for each level of theory given
        a list of CBH.buildCBH objects or a dictionary containing 
        reactions.

        ARGUMENTS
        ---------
        :s:             [str] 
            SMILES string of the target species.
        
        :cbh_or_altrxn: [list(CBH.buildCBH) or dict] 
            CBH.buildCBH object or dictionary containing reactions.
            If dict, it must follow this format: 
                {rank : {s: {rxn_dict}}}

        :skip_precursor_check:  [bool] (default=False)
            Whether skip the procedure to check if precursors exist in 
            database or if they have real values. Suggested to skip 
            only when the cbh_rung is validated after 
            check_rung_usability method is used.

        :cbh_rungs:     [list] (default=None) 
            List of CBH rungs to weight. Supply if cbh_or_altrxn 
            is a list of CBH.buildCBH objects.

        RETURNS
        -------
        weighted_Hrxn:       [dict] Weighted Hrxn values for each level of theory.

        weighted_Hf:         [dict] Weighted Hf values for each level of theory.
        """

        if isinstance(cbh_or_altrxn, list):
            for cbh in cbh_or_altrxn:
                if not isinstance(cbh, CBH.buildCBH):
                    raise TypeError(f'List provided to function must contain CBH.buildCBH objects. \nType {type(cbh)} was provided instead.')
            if not isinstance(cbh_or_altrxn, list):
                raise ValueError(f'Since arg: cbh_or_altrxn was a list of CBH.buildCBH objects, a list of CBH rungs must be provided to the arg: cbh_rungs.')

        elif isinstance(cbh_or_altrxn, dict):
            # TODO: Might need an additional check
            pass

        # compute Hrxn and Hf
        # if only one saturation
        if len(cbh_or_altrxn) == 1:
            if isinstance(cbh_or_altrxn, list):
                rxn = self._cbh_to_rxn(s, cbh_or_altrxn[0], cbh_rungs[0])
            elif isinstance(cbh_or_altrxn, dict):
                rxn = cbh_or_altrxn[list(cbh_or_altrxn.keys())[0]]

            weighted_Hrxn, weighted_Hf = self.Hf(s, rxn, skip_precursor_check=skip_precursor_check)
            weighted_Hrxn = {k: abs(v) for k, v in weighted_Hrxn.items()}
        # Weight multiple reactions
        else:
            Hrxn_ls = []
            Hf_ls = []
            if isinstance(cbh_or_altrxn, list):
                iterable_obj = cbh_or_altrxn
            elif isinstance(cbh_or_altrxn, dict):
                iterable_obj = cbh_or_altrxn.values()
            for i, rxn in enumerate(iterable_obj):
                if isinstance(cbh_or_altrxn, list):
                    rxn = self._cbh_to_rxn(s, rxn, cbh_rungs[i])
                Hrxn_s, Hf_s = self.Hf(s, rxn, skip_precursor_check=skip_precursor_check)
                Hrxn_ls.append(Hrxn_s)
                Hf_ls.append(Hf_s)

            weights = {}
            for k in Hrxn_ls[0]:
                weights[k] = self._weight(*[Hrxn_s[k] for Hrxn_s in Hrxn_ls])
            # weights = {method_keys : [weights_cbh_type]}
            weighted_Hrxn = {}
            weighted_Hf = {}
            for k in weights.keys():
                weighted_Hrxn[k] = sum([weights[k][i]*abs(Hrxn_s[k]) for i, Hrxn_s in enumerate(Hrxn_ls)])
                weighted_Hf[k] = sum([weights[k][i]*Hf_s[k] for i, Hf_s in enumerate(Hf_ls)])
            
        return weighted_Hrxn, weighted_Hf

    
    def calc_Hf_from_source_vectorized(self, s:str, DfH_UQ_array:np.array, DfH_UQ_index:pd.Index, source_str:str=None):
        """
        Calculate Hrxn and Hf of a species, s, from the 'source' column in 
        self.energies or a provided source_str. Primarily used for 
        UQ.uncertainty_quantification.

        ASSUMES ALL REACTANTS RELATED TO THE SOURCE EXIST

        ARGUMENTS
        ---------
        :s:             [str] 
            SMILES string of the target species.

        :DfH_UQ_array:  [np.array] shape=(num_species, num_simulations)
            2D array holding the sampled DfH values for each species.
        
        :DfH_UQ_index:  [pd.Index] 
            List of species names corresponding to the rows of DfH_UQ_array.

        :source_str:    [str] (optional, default=None) 
            String for CBH/alternative rung to use.
            ex) 'CBH-#-S' 
                (where # is an int/float of the rung and S is the saturation
                    atom or "alt" for alternative reaction)
            ex) 'CBHavg-(#-S, #-S, ...)'
                (average of different rungs, includes alternative rxns)
        
        RETURNS
        -------
        (weighted_Hrxn, weighted_Hf)

        :weighted_Hrxn: [float] Weighted heat of reaction based on qchem values.

        :weghted_Hf:    [1D np.array] shape=(1, num_simulations)
            Heat of formation for each simulation. num_simulations corresponds to 
            second dimention of DfH_UQ_array.
        """

        if source_str is None:
            source_str = self.energies.loc[s, 'source']

        rungs = []
        sats = []
        if 'CBH' in source_str:
            if 'avg' in source_str.split('//')[0]:
                # ex. CBHavg-(#-S, #-S, #-alt)
                rungs += [float(sub.split('-')[0]) for sub in source_str.split('//')[0][8:-1].split(', ')]
                sats += [sub.split('-')[1] for sub in source_str.split('//')[0][8:-1].split(', ')]
            else:
                # ex. CBH-#-S
                rungs += [float(source_str.split('//')[0].split('-')[1])]
                sats += [source_str.split('//')[0].split('-')[2]]
            
            methods = source_str.split('//')[1].split('+')
        else:
            # if not from CBH rung --> experimental
            methods = source_str.split('+')

        rxns = {}
        for i, sat in enumerate(sats):
            if sat != 'alt':
                cbh = CBH.buildCBH(s, sat, allow_overshoot=True, surface_smiles=self.surface_smiles)
                rxns[i] = self._cbh_to_rxn(s, cbh, rungs[i])
            else:
                rxns[i] = {s: copy(self.alternative_rxn[s][rungs[i]])}
                rxns[i][s].update({s:-1})
        
        nrg_cols = ['DfH']
        for method in methods:
            nrg_cols.extend(self.methods_keys_dict[method])

        Hrxn = {}
        Hf = {}
        for r, rxn in rxns.items():
            # heat of rxn
            rxn_species = list(rxn[s].keys())
            species_ind_for_df = DfH_UQ_index.get_indexer(rxn_species)

            coeff_vec = np.array(list(rxn[s].values())) * -1 # products are negative, and reactants are positive
            nrg_arr = self.energies.loc[rxn_species, nrg_cols].values
            matrix_mult = np.matmul(coeff_vec, nrg_arr) # for Hrxn

            DfH_all_simulations = coeff_vec @ DfH_UQ_array[species_ind_for_df,:] # - self.energies.loc[s, 'DfH']

            delE = {nrg_cols[i]:matrix_mult[i] for i in range(len(nrg_cols))}
            Hrxn[r] = {**dict(zip(methods, np.full(len(methods), nan)))}
            Hf[r] = np.tile(DfH_all_simulations*-1, (len(methods), 1))

            # in most cases, it is just summing the values in method_keys for a given method
            for m, method in enumerate(methods):
                rank = self.rankings_rev[method]
                # This is assuming rank=1 is experimental and already measures Hf
                if rank == 0 or rank == 1:
                    continue
                elif method == 'anl0' and True not in self.energies.loc[rxn[s].keys(), ['avqz', 'av5z', 'zpe']].isna():
                    Hrxn[r]['anl0'] = anl0_hrxn(delE)
                elif 0 not in self.energies.loc[rxn[s].keys(), self.methods_keys_dict[method]].values:
                    Hrxn[r][method] = sum_Hrxn(delE, *self.methods_keys_dict[method])

                Hf[r][m, :] += Hrxn[r][method]
        
        # Hrxn =    {rxn : {method : DrxnH}}
        # weights = {method : [weights in order of rxn]}
        # Hf =      {rxn : 2D array (num_methods, num_simulations)}

        weights = {}
        # 1. get weights for each rxn 
        for method in methods:
            weights[method] = self._weight(*[Hrxn[r][method] for r in Hrxn.keys()])

        # 2. apply weights to get Hrxn = {method : avg DrxnH} and Hf
        weighted_Hrxn = {}
        for m, method in enumerate(methods):
            weighted_Hrxn[method] = sum([weights[method][r]*abs(Hrxn[r][method]) for r in Hrxn.keys()])
            for r in Hf.keys():
                Hf[r][m,:] = weights[method][r] * Hf[r][m,:]

        weighted_Hf = np.zeros(Hf[r].shape)
        for r in Hf.keys():
            weighted_Hf += Hf[r]

        # 3. get weights for each method
        weights = self._weight(*[v for v in weighted_Hrxn.values()])

        # 4. apply weights to get Hrxn and Hf
        for i, w in enumerate(weights):
            weighted_Hf[i,:] = w * weighted_Hf[i,:]
        
        weighted_Hrxn = sum([weights[i]*weighted_Hrxn[method] for i, w in enumerate(list(weighted_Hrxn.keys()))])
        weighted_Hf = np.sum(weighted_Hf, axis=0)
        
        return weighted_Hrxn, weighted_Hf


    def calc_Hf_allrungs(self, s: str, saturate: int or str=1, surface_smiles:str=None) -> tuple:
        """
        Calculates the Hf of a given species at each CBH rung.
        Assumes that calc_Hf has already been run or the database
        has all of the necessary energies of molecules.

        ARGUMENTS
        ---------
        :s:         [str] SMILES string of the target species.

        :saturate:  [int or str] (default=1)
                Atomic number of saturation element.

        :surface_smiles:    [str] (default=None)

                Valid SMILES string representing the surface atom that the given 
                molecule is adsorbed to or physiosorbed to. Must be a single atom.
                i.e., '[Pt]'

        RETURN
        ------
        (Hrxn, Hf): [tuple]
            Hrxn    [dict] {rung : heat of reaction}
            Hf      [dict] {rung : heat of formation}
        """
        
        ptable = GetPeriodicTable()
        if type(saturate) == str:
            saturate = ptable.GetAtomicNumber(saturate)
        saturate_sym = ptable.GetElementSymbol(saturate)

        s = CanonSmiles(s) # standardize SMILES
        s_cbh = CBH.buildCBH(s, saturate, surface_smiles=surface_smiles)

        Hf = {}
        Hrxn = {}
        if s in self.error_messages:
            start_err_len = len(self.error_messages[s])
        else:
            start_err_len = 0

        for rung in range(s_cbh.highest_cbh+1):
            try:
                prepend_str = 'Computing all heats of formation at all rungs. \nErrors below are reflected by this operation'
                if s not in self.error_messages:
                    self.error_messages[s] = []
                    self.error_messages[s].append(prepend_str)

                test_rung = self._check_rung_usability(s, rung, s_cbh.cbh_rcts, s_cbh.cbh_pdts, saturate_sym)
                
                rxn = self._cbh_to_rxn(s, s_cbh, rung)
                
                check_precursors_exist = [p in self.energies.index.values for p in rxn[s].keys()]
                if all(check_precursors_exist):
                    Hrxn[rung], Hf[rung] = self.Hf(s, rxn, skip_precursor_check=True)
                else:
                    continue

            except TypeError:
                raise TypeError(f'Cannot compute CBH-{rung}')

        if self.error_messages[s][-1] == prepend_str:
            if len(self.error_messages[s])==1:
                del self.error_messages[s]
            else:
                self.error_messages[s].pop() # delete prepend_str
        else:
            self.error_messages[s].append('Conclude computation of all heats of formation for all rungs.')

        if s in self.error_messages and start_err_len > len(self.error_messages[s]):
            print(f'Errors encountered while computing all heats of formation for {s}. View with error_messages[{s}] methods.')

        return Hrxn, Hf


    def _choose_best_method(self, Hrxn: dict, Hf: dict, label: str):
        """
        Chooses the best level of theory to log in self.energies dataframe.

        ARGUMENTS
        ---------
        :Hrxn:      [dict] holds heat of reaction for each level of theory
                        {theory : H_rxn}
        :Hf:        [dict] holds heat of formation for each level of theory
                        {theory : H_f}
        :label:     [str] String denoting CBH level - prefix for 'source'

        RETURNS
        -------
        (weighted_Hrxn, weighted_Hf, new_label)

        :weighted_Hrxn:     [float] heat of reaction for best level of theory
        :weighted_Hf:       [float] heat of formation for best level of theory
        :new_label:         [str] label to be used for self.energies['source']
        """

        # Choose the best possible method to assign to the energies dataframe
        if all([isnan(v) for v in Hrxn.values()]):
            # if all computed heats of reactions are NaN
            return nan, nan, nan

        else:
            # list of all levels of theory in Hf and Hrxn
            hrxn_theories = set(list(Hrxn.keys()))

            # cyle through ascending rank
            for rank in set(list(self.rankings.keys())):
                # record theories in Hrxn dict for a given rank
                theories_in_hrxn = [theory for theory in set(self.rankings[rank]) if theory in hrxn_theories]

                if rank == 0 or len(theories_in_hrxn) == 0:
                    # we ignore rank=0
                    # or go to next rank if none exist for current one
                    continue 

                elif len(theories_in_hrxn) == 1:
                    # only one level of theory in this rank
                    theory = theories_in_hrxn[0]

                    # ensure computed Hrxn is not NaN
                    if not isnan(Hrxn[theory]):
                        return Hrxn[theory], Hf[theory], label+'//'+theory
                    else:
                        # go to lower rank
                        continue
                
                # when multiple equivalent rankings
                elif len(theories_in_hrxn) >= 1:
                    theories = []
                    hrxns = []
                    hfs = []

                    for theory in theories_in_hrxn:
                        if not isnan(Hrxn[theory]):
                            theories.append(theory)
                            hrxns.append(Hrxn[theory])
                            hfs.append(Hf[theory])
                        else:
                            continue
                    
                    # in case all are nan, go to lower rank
                    if len(hrxns) == 0:
                        continue
                    
                    # in case only 1 was not nan
                    elif len(hrxns) == 1:
                        return hrxns[0], hfs[0], label + '//' + theories[0]
                    
                    else:
                        # generate weighted Hrxn and Hf
                        rank_weights = np.array(self._weight(*hrxns))
                        weighted_hrxn = np.sum(np.array(hrxns) * rank_weights)
                        weighted_hf = np.sum(np.array(hfs) * rank_weights)

                        # create new label that combines all levels of theory
                        new_label = '' + label
                        for i, t in enumerate(theories):
                            if i == 0:
                                new_label += '//'+t
                            else:
                                new_label += '+'+t

                        return weighted_hrxn, weighted_hf, new_label

        return nan, nan, nan


    def _weight(self, *Hrxn: float):
        """
        Weighting for combining different CBH schemes at the same level.
        w_j = |∆Hrxn,j|^-1 / sum_k(|∆Hrxn,k|^-1)

        ARGUMENTS
        ---------
        :Hrxn:      [float] Heat of reactions to weight

        RETURNS
        -------
        :weights:   [list] list of weights (sum=1)
        """
        weights = np.zeros((len(Hrxn)))

        # if Hrxn is 0 for any species, return one-hot encoding
        if 0 in list(Hrxn):
            Hrxn = np.array(list(Hrxn))
            ind = np.where(Hrxn==0)[0]

            weights[ind] = 1 / len(ind)
            return weights.tolist()
        
        # more common case where Hrxn is not exactly 0
        denom = 0
        # calculate denom
        for h in Hrxn:
            denom += (1/abs(h))

        # calculate weights
        for i, h in enumerate(Hrxn):
            weights[i] = 1/abs(h) / denom

        return weights.tolist()

    
    def _check_rung_usability(self, s: str, test_rung: int, cbh_rcts: dict, cbh_pdts: dict, label: str): 
        """
        Method that checks a given CBH rung's usability by looking for missing 
        reactants or whether all precursors were derived using CBH schemes of 
        the same level of theory. Errors will cause the method to move down to
        the next rung and repeat the process. It stores any errors to the 
        calcCBH.error_messages attribute which can be displayed with the 
        calcCBH.print_errors() method. 

        If all species in a CBH rung (including the target) have heats of formation
        derived using CBH and with homogeneous levels of theory, rung equivalency 
        will be checked after decomposing the reaction such that all precursors are 
        derived using "reference" levels of theory. "Reference" refers to a theory
        that is of better rank than the best possible theory for the target species, 
        s.

        ARGUMENTS
        ---------
        :s:         [str] SMILES string of the target species
        :test_rung: [int] Rung number to start with. (usually the highest 
                        possible rung)
        :cbh_rcts:  [dict] The target species' CBH.cbh_rcts attribute
                        {rung # : [reactant SMILES]}
        :cbh_pdts:  [dict] The target species' CBH.cbh_pdts attribute
                        {rung # : [product SMILES]}
        :label:     [str] The label used for the type of CBH
                        ex. 'H' for hydrogenated, 'F' for fluorinated

        RETURNS
        -------
        test_rung   [int] The highest rung that does not fail
        """

        for rung in reversed(range(test_rung+1)):
            # 1. check existance of all reactants in database
            #   a. get precursors in a rung
            #   b. get the sources of those precursors
            all_precursors = list(cbh_rcts[rung].keys()) + list(cbh_pdts[rung].keys())
            try:
                sources = self.energies.loc[all_precursors, 'source'].values.tolist()
                # if all values contributing to the energy of a precursor are NaN move down a rung
                species_null = self.energies.loc[all_precursors, self.methods_keys].isna().all(axis=1)
                if True in species_null.values:
                    test_rung -= 1
                    # error message
                    missing_precursors = ''
                    for precursors in species_null.index[species_null.values]:
                        missing_precursors += '\n\t   '+precursors
                    self.error_messages[s].append(f"CBH-{test_rung+1}-{label} precursor(s) do not have any calculations in database: {missing_precursors}")
                    if test_rung >= 0:
                        self.error_messages[s][-1] += f'\n\t Rung will move down to {test_rung}.'
                    continue
            
            # triggers when a precursor in the CBH scheme does not appear in the database
            except KeyError as e:
                test_rung -= 1
                
                # find all precursors that aren't present in the database
                missing_precursors = ''
                for precursors in all_precursors:
                    if precursors not in self.energies.index:
                        missing_precursors += '\n\t   '+precursors

                self.error_messages[s].append(f"CBH-{test_rung+1}-{label} has missing reactant(s): {missing_precursors}")
                if test_rung >=0:
                    self.error_messages[s][-1] += f'\n\tRung will move down to CBH-{test_rung}-{label}.'
                continue

            # 2. check sources of target and all reactants
            #   2a. Get the highest rank for level of theory of the target
            target_energies = self.energies.loc[s, self.methods_keys]
            for rank in set(self.rankings.keys()):
                if rank==0: # don't consider rank==0
                    continue

                avail_theories = [theory for theory in set(self.rankings[rank]) if theory in set(self.methods_keys_dict.keys())]
                if len(avail_theories) == 0:
                    continue
                else:
                    for theory in avail_theories:
                        # check for NaN in necessary keys to compute given theory
                        if True not in target_energies[self.methods_keys_dict[theory]].isna().values:
                            break
                        else:
                            # if nan, try next theory
                            continue
                    else: # triggers when no break occurs (energy vals in avail_theories were nan)
                        continue
                    # if a break occurs in the inner loop, break the outer loop
                    break
            
            #   2b. Get list of unique theories for each of the precursors
            if False not in [type(s)==str for s in sources]:
                new_sources = [s.split('//')[1].split('+')[0] if 'CBH' in s and len(s.split('//'))>1 else s for s in sources]
                source_rank = list([self.rankings_rev[source] for source in new_sources])
            else: # if a species Hf hasn't been computed yet, its source will be type==float, so move down a rung
                # This will appropriately trigger for overshooting cases
                test_rung -= 1
                continue
            
            # reaction dictionary
            rxn = add_dicts(cbh_pdts[rung], {k : -v for k, v in cbh_rcts[rung].items()}) 

            # 3. Decompose precursors until all are a better rank than the target
            # delete "len(set(source_rank))==1 and " ?????
            if max(source_rank) >= rank and max(source_rank) != 1:
                verbose_error = False
                while max(source_rank) >= rank:                    
                    if max(source_rank) > rank:
                        # This leaves potential to get better numbers for the precursors with larger ranks.
                        err_precur = list(compress(all_precursors, [r > rank for r in source_rank]))
                        missing_precursors = self._missing_precursor_str(err_precur)
                        self.error_messages[s].append(f"A precursor of the target molecule was computed with a level of theory that is worse than the target. \nConsider recomputing for: {missing_precursors}")

                    rxn_sources = self.energies.loc[list(rxn.keys()), 'source'].values.tolist() # sources of precursors in reaction
                    named_sources = [s.split('//')[1].split('+')[0] if 'CBH' in s and len(s.split('//'))>1 else s for s in rxn_sources]
                    # indices where theory is worse than rank
                    worse_idxs = np.where(np.array([self.rankings_rev[s] for s in named_sources]) >= rank)[0]
                    decompose_precurs = [list(rxn.keys())[i] for i in worse_idxs] # precursors where the theory is worse than rank and must be decomposed
                    # dictionaries holding a precursor's respective rung rung or saturation atom

                    d_rung = {}
                    d_sat = {}
                    for i, precur in enumerate(list(rxn.keys())):
                        if len(rxn_sources[i].split('//')) > 1:
                            if 'avg' in rxn_sources[i].split('//')[0]:
                                # ex. CBHavg-(N-S, N-S, N-alt)
                                d_rung[precur] = [float(sub.split('-')[0]) for sub in rxn_sources[i].split('//')[0][8:-1].split(', ')]
                                d_sat[precur] = [sub.split('-')[1] for sub in rxn_sources[i].split('//')[0][8:-1].split(', ')]
                            else:
                                # ex. CBH-N-S
                                d_rung[precur] = [float(rxn_sources[i].split('//')[0].split('-')[1])]
                                d_sat[precur] = [rxn_sources[i].split('//')[0].split('-')[2]]
                        else:
                            # shouldn't happen
                            pass
                    
                    for precur in decompose_precurs:
                        if precur not in rxn.keys():
                            continue
                        # choose saturation / rung to check
                        if len(d_sat[precur]) == 1:
                            p_sat = d_sat[precur][0]
                            p_rung = d_rung[precur][0]
                        elif label in d_sat[precur]: 
                            p_sat = label
                            # if multiple 'alt' reactions, this could be an issue
                            p_rung_idx = d_sat[precur].index(label)
                            p_rung = d_rung[precur][p_rung_idx]
                        else:
                            # Choose the rxn with the lowest rung number
                            # Lowest rung number --> usually smaller species with exp / better theoretical values
                            idx_lowest_rung = np.where(np.array(d_rung[precur]) == min(d_rung[precur]))[0][0]
                            p_sat = d_sat[precur][idx_lowest_rung]
                            p_rung = d_rung[precur][idx_lowest_rung]

                        if p_sat != 'alt':

                            p = CBH.buildCBH(precur, p_sat, allow_overshoot=True, surface_smiles=self.surface_smiles)
                            try: 
                                self.energies.loc[list(p.cbh_pdts[p_rung].keys()) + list(p.cbh_rcts[p_rung].keys())]
                                # To the original equation, add the precursor's precursors (rcts + pdts) and delete the precursor from the orig_rxn
                                rxn = add_dicts(rxn, {k : rxn[precur]*v for k, v in p.cbh_pdts[p_rung].items()}, {k : -rxn[precur]*v for k, v in p.cbh_rcts[p_rung].items()})
                                del rxn[precur]
                            except KeyError:
                                # new precursor does not exist in database
                                dont_exist = [pp for pp in p.cbh_pdts[p_rung].keys() if pp in self.energies.index.values]
                                dont_exist += [pp for pp in p.cbh_rcts[p_rung].keys() if pp in self.energies.index.values]
                                self.error_messages[s].append(f"Error occurred during decomposition of CBH-{test_rung}-{label} when checking for lower rung equivalency. \n\tSome reactants of {precur} did not exist in the database.")
                                self.error_messages[s][-1] += f"\n\t\tThese reactants were: {dont_exist}"
                                self.error_messages[s][-1] += f"\n\tThere is a possibility that CBH-{test_rung}-{label} of {s} \n\tis equivalent to a lower rung, but this cannot be rigorously tested automatically."
                                verbose_error = True
                                break
                        else:
                            if set(self.alternative_rxn[precur][p_rung].keys()).issubset(self.energies.index.values):
                                rxn = add_dicts(rxn, {rct: coeff*rxn[precur] for rct, coeff in self.alternative_rxn[precur][p_rung].items()})
                                del rxn[precur]
                            else:
                                self.error_messages[s].append(f"Error occurred during decomposition of CBH-{test_rung}-{label} when checking for lower rung equivalency. \n\tSome reactants of {precur} did not exist in the database.")
                                self.error_messages[s][-1] += f"\n\tThere is a possibility that CBH-{test_rung}-{label} of {s} \n\tis equivalent to a lower rung, but this cannot be rigorously tested automatically."
                                verbose_error = True
                                break
                    
                    else: # if the for loop doesn't break (most cases)
                        rxn_sources = self.energies.loc[list(rxn.keys()), 'source'].values.tolist()
                        
                        if False not in [isinstance(s, str) for s in rxn_sources]:
                            check_new_sources = [s.split('//')[1].split('+')[0] if 'CBH' in s and len(s.split('//'))>1 else s for s in rxn_sources]
                        else:
                            print('Uh Oh')
                            # Uh Oh
                            pass

                        source_rank = list([self.rankings_rev[s] for s in check_new_sources])
                        continue # continue while loop
                    
                    # will only trigger if the for-loop broke
                    break
                
                # Check whether the decomposed reaction is equivalent to other rungs
                rung_equivalent, equiv_rung = self._check_rung_equivalency(s, rung, cbh_rcts, cbh_pdts, rxn, verbose_error)
                
                if rung_equivalent:
                    # error message
                    self.error_messages[s].append(f'All precursor species for CBH-{test_rung}-{label} had the same level of theory. \n\tThe reaction was decomposed into each precursors\' substituents which was found to be equivalent to CBH-{equiv_rung}-{label}.\n\tMoving down to CBH-{test_rung-1}-{label}.')
                    test_rung -= 1
                    continue
                else:
                    return test_rung
            else:
                return test_rung
            
        return test_rung

    
    def _check_rung_equivalency(self, s:str, top_rung: int, cbh_rcts: dict, cbh_pdts: dict, precur_rxn: dict, verbose_error=False):
        """
        Helper function to check reaction equivalency. 
        Add the dictionaries for the precursor total reaction with the
        CBH products and negated CBH reactants of the target. Equivalent 
        reactions will yield values of 0 in the output dictionary.

        ARGUMENTS
        ---------
        :s:                 [str] SMILES string of the target species
        :top_rung:          [int] Rung that is currently being used for CBH of 
                                target. The function will check rungs lower than 
                                this (not including).

        :cbh_rcts:          [dict] Reactant precursors of the target molecule 
                                for each CBH rung.
                                {CBH_rung : {reactant_precursor : coeff}}
        :cbh_pdts:          [dict] Product precursors of the target molecule 
                                for each CBH rung.
                                {CBH_rung : {product_precursor : coeff}}
        :decomposed_rxn:    [dict] A decomposed CBH precursor dictionary
                                {reactant/product_precursor : coeff}
        :verbose_error:     [bool] (default=False) Show equivalency dictionaries
                                in self.error_messages.

        RETURNS
        -------
        (rung_equivalent, rung)    
            :rung_equivalent:      [bool] False if no matches, True if 
                                    equivalent to lower rung
            :rung:                 [int] Rung at given or equivalent rung
        """
        if verbose_error:
            self.error_messages[s][-1] += "\n\tReaction dictionaries hold the target species' precursors and respective coefficients \n\twhere negatives imply reactants and positives are products."
            self.error_messages[s][-1] += f"\n\tBelow is the partially decomposed reaction: \n\t{precur_rxn}"
            self.error_messages[s][-1] += f"\n\tThe decomposed reaction will be subtracted from each CBH rung. \n\tFor equivalency, all coefficients must be 0 which would yield an emtpy dictionary."

        for r in reversed(range(top_rung)): # excludes top_rung
            new_cbh_pdts = {k : -v for k, v in cbh_pdts[r].items()}
            total_prec = add_dicts(new_cbh_pdts, cbh_rcts[r], precur_rxn)
            if verbose_error:
                self.error_messages[s][-1] += f"\n\tSubtracted from CBH-{r}: \n\t{total_prec}"
            
            if not total_prec: # empty dict will return False
                if verbose_error and 'Subtracted from' in self.error_messages[s][-1]:
                    # this deletes the verbose nature of the error message since it already prints that the reaction is equivalent to a lower rung
                    del self.error_messages[s][-1]
                return True, r

        return False, top_rung


    def _check_alt_rxn_usability(self, s:str, alt_rxn_keys:list):
        """
        Helper method to check whether the species present in the provided
        alternative reaction(s) exist in the database.

        ARGUMENTS
        ---------
        :s:             [str] Species SMILES string

        :alt_rxn_keys:  [list] List of keys defining the alternative reactions 
                            to check for the species.

        RETURN
        ------
        alt_rxns [dict or None] Returns usable alt_rxns dictionary. Otherwise
                        returns None.
                        alt_rxns = {alt_rank : {s : {alt_rxn_species : coeff}}}
        """
        alt_rxns = {}
        for alt_rank in alt_rxn_keys:
            alt_rxns[alt_rank] = {}
            alt_rxns[alt_rank][s] = copy(self.alternative_rxn[s][alt_rank])
            alt_rxns[alt_rank][s][s] = -1

            # check alternative rxn precursors existance in database
            species = list(alt_rxns[alt_rank][s].keys())
            if set(species).issubset(self.energies.index.values):
                species_null = self.energies.loc[species, self.methods_keys].isna().all(axis=1)
                # check ∆Hf
                if True in self.energies.loc[species, 'source'].isna():
                    m_species = [species[i] for i, cond in enumerate(self.energies.loc[species, 'source'].isna().values) if cond==True]
                    missing_precursors = self._missing_precursor_str(m_species)
                    self.error_messages[s].append(f'The following precursors did not have usable reference heats of formation for the provided alternative reaction: {missing_precursors}')
                    del alt_rxns[alt_rank]
                    self.error_messages[s][-1] += '\nTrying remaining alternative rung(s) instead.'

                # check QM energies
                elif True in species_null:
                    missing_precursors = self._missing_precursor_str(species_null.index[species_null.values])
                    self.error_messages[s].append(f'The following precursors did not have usable QM values for the provided alternative reaction: {missing_precursors}')
                    del alt_rxns[alt_rank]
                    self.error_messages[s][-1] += '\nTrying remaining alternative rung(s) instead.'

            else:
                m_species = [sp for sp in species if sp not in self.energies.index.values and sp != s]
                missing_precursors = self._missing_precursor_str(m_species)
                self.error_messages[s].append(f'Precursors in provided alternative reaction did not exist in database: {missing_precursors}')
                self.error_messages[s][-1] += '\nUtilizing CBH scheme instead.'
                del alt_rxns[alt_rank]
                self.error_messages[s][-1] += '\nTrying remaining alternative rung(s) instead.'

        if len(alt_rxns) != 0:
            return alt_rxns
        else:
            # when no alternative reactions are usable
            self.error_messages[s][-1] += '\nUtilizing CBH scheme instead.'
            return None

    
    def _missing_precursor_str(self, missing_precursors):
        """
        Generate string for self.error_messages that formats all 
        precursors in arg: missing_precursors.

        ARGUMENTS
        ---------
        :missing_precursors:    [list] list of precursors that belong
                                    in self.error_messages.

        RETURNS
        -------
        :missing_precursor_str: [str] formated string
        """
        
        missing_precursors_str = ''
        for precursors in missing_precursors:
            missing_precursors_str += '\n\t   '+precursors
        return missing_precursors_str
    
    
    def print_errors(self):
        """
        Print error messages after calcCBH.calcHf() method completed.

        ARGUMENTS
        ---------
        None

        RETURNS
        -------
        None
        """
        if len(list(self.error_messages.keys())) != 0:
            for s in self.error_messages.keys():
                print(f'{s}:')
                for m in self.error_messages[s]:
                    print('\n\t'+m)
                print('\n')
        else:
            print('No errors found.')

    
    def save_calculated_Hf(self, save_each_molecule_file:bool=False, save_pd_dictionary:bool=False, **kwargs):
        """
        THIS ACTION IS IRREVERSIBLE
        ---------------------------
        
        Save calculated Hf and Hrxn to database. 
        

        ARGUMENTS
        ---------
        :save_each_molecule_file:   [bool]
            Save the newly calculated heats of formation to each molecule's 
            file in the folder given by parameter: folder_path

            REQUIRES ADDITIONAL KWARG
            :folder_path:       [str] Path to a folder that contains the files 
                                    of each molecule

        :save_pd_dictionary:        [bool]
            Save the self.energies dataframe with updated values to 
            a pickle file after converting to a dictionary.

            REQUIRES ADDITIONAL KWARG
            :file_path:       [str] Filepath to save self.energies DataFrame as 
                                    a pickled dictionary. Must end in '.pkl'
        
        RETURNS
        -------

        """
        for k, v in kwargs.items():
            if not isinstance(v, str):
                raise TypeError(f'Argument {k} must be of type str, not {type(v)}')

        if save_each_molecule_file:
            try:
                folder_path = kwargs['folder_path']
            except KeyError as e:
                # err_msg = KeyErrorMessage()
                raise KeyError('Since save_each_molecule_file was True, the user must provide a path to a folder that contains the files of each molecule.')

            for filename in os.listdir(folder_path):
                f = os.path.join(folder_path, filename)
                if os.path.isfile(f):
                    with open(f, 'r') as yamlfile:
                        yamldict = yaml.safe_load(yamlfile)

                        smiles = CanonSmiles(yamldict['smiles'])
                        source = self.energies.loc[smiles, 'source']
                        hf = self.energies.loc[smiles, 'DfH']
                        hrxn = self.energies.loc[smiles, 'DrxnH']

                        save = False
                        # heat of formation
                        if 'heat_of_formation' in yamldict:
                            if source in yamldict['heat_of_formation']:
                                
                                if abs(yamldict['heat_of_formation'][source] - hf) > 1e-6:
                                    # this if statement is unstable for some reason
                                    save = True
                                    print(f'Rewriting heat of formation for:\n\tFile path: \t{f}\n\tPrevious value: \t {yamldict["heat_of_formation"][source]} \n\tNew value: \t{hf}\n')
                            yamldict['heat_of_formation'][source] = float(hf)

                        else:
                            save = True
                            yamldict['heat_of_formation'] = {source : float(hf)}
                        
                        # heat of reaction
                        if 'heat_of_reaction' in yamldict:
                            if source in yamldict['heat_of_reaction']:
                                if yamldict['heat_of_reaction'][source] != hrxn:
                                    print(f'Rewriting heat of reaction for:\n\tFile path: \t{f}\n\tPrevious value: \t {yamldict["heat_of_reaction"][source]} \n\tNew value: \t{hrxn}\n')
                            yamldict['heat_of_reaction'][source] = float(hrxn)
                        else:
                            yamldict['heat_of_reaction'] = {source : float(hrxn)}
                        
                        # update yaml file
                        if save:
                            with open(f, 'w') as yamlfile:
                                yaml.safe_dump(yamldict, yamlfile, default_flow_style=False)

        if save_pd_dictionary:
            try:
                file_path = kwargs['file_path']
            except KeyError as e:
                raise KeyError('Since save_pd_dictionary was True, the user must provide a filepath for which to save the self.energies dataframe.')

            # check last 5 in str and make sure it's '.pkl', else add it
            if file_path[-4:] != '.pkl':
                raise NameError(f'The filepath: {file_path} must end in ".pkl"')

            # create directory if directory doesn't exist.
            folder_path = file_path.split('.pkl')[0].split('/')[:-1]
            if not os.path.exists(os.path.join(*folder_path)):
                os.makedirs(os.path.join(*folder_path), exist_ok=True)
            
            self.energies.to_pickle(file_path)
            print(f'Saved to: {file_path}')


    @staticmethod
    def generate_CBH_coeffs(species_list: list, saturate: int=1, allow_overshoot=False, surface_smiles:str=None, include_target=True) -> list:
        """
        Generate a list of Pandas DataFrame objects that hold the coefficients 
        for every precursor created for CBH schemes of each target species.

        ARGUMENTS
        ---------
        :species_list:      [list] List of SMILES strings with target molecules.

        :saturate:          [int or str] Integer representing the atomic number 
                                of the element to saturate residual species with. 
                                String representation of the element also works.
                                (default=1)

        :allow_overshoot:   [bool] (default=False)
                                Choose to allow a precursor to have a substructure 
                                of the target species, as long as the explicit 
                                target species does not show up on the product 
                                side. 
                Ex) if species='CC(F)(F)F' and saturate=9, the highest CBH rung 
                    would be: 
                    False - CBH-0-F
                    True - CBH-1-F (even though C2F6 is generated as a precursor)

        :surface_smiles:    [str] (default=None)
                Valid SMILES string representing the surface atom that the given 
                molecule is adsorbed to or physiosorbed to. Must be a single atom.
                i.e., '[Pt]'

        :include_target:    [bool] (default=True) 
                Include the target species with a coefficient of -1

        RETURNS
        -------
        :dfs:   [list] List of DataFrames for each rung. Each DataFrame holds 
                    the coefficients of precursors for each target species.

        Example
        -------
        dfs = [df0, df1, df2, ..., dfn] Where n = the highest CBH rung
        
        dfn = DataFrame where the index is the target species, and columns are 
                precursor species

        Data within DataFrame are coefficients for precursor species that are 
        used for the CBH rung of the given target species.
        """
        
        species_list = [CanonSmiles(species) for species in species_list] # list of SMILES strings

        # initialize dictionaries to hold each species' CBH scheme
        all_rcts = {} # {species: {CBH scheme reactants}}
        all_pdts = {} # {species: {CBH scheme products}}
        highest_rung_per_molec = []
        rm_species = []
        for species in species_list:
            try:
                cbh = CBH.buildCBH(species, saturate, allow_overshoot=allow_overshoot, surface_smiles=surface_smiles) # generate CBH scheme
            except:
                print(f'Cannot compute CBH for "{species}". Continuing without it.')
                rm_species.append(species)
                continue
            # add to dictionary / lists
            all_rcts[species] = cbh.cbh_rcts
            all_pdts[species] = cbh.cbh_pdts
            highest_rung_per_molec.append(cbh.highest_cbh)
        
        # Find the highest CBH rung of all the CBH schemes
        highest_rung = max(highest_rung_per_molec)

        species_list = [s for s in species_list if s not in rm_species]
        dfs = [] # initialize DataFrame list
        # Cycle through each CBH rung
        for rung in range(highest_rung+1):
            df = {}
            # Cycle through each species in species_list
            for species in species_list:
                if rung <= max(all_pdts[species].keys()):
                    df[species] = all_pdts[species][rung]
                    df[species].update((precursor, coeff * -1) for precursor,coeff in all_rcts[species][rung].items())
                    if include_target:
                        df[species].update({species:-1})
            dfs.append(pd.DataFrame(df).fillna(0).T)
        return dfs