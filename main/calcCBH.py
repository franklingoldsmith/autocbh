import CBH
from CBH import add_dicts
import numpy as np
import pandas as pd
from numpy import nan, isnan
from rdkit.Chem import MolFromSmiles, AddHs, CanonSmiles, GetPeriodicTable
from data.molData import load_rankings, generate_database, generate_alternative_rxn_file
from hrxnMethods import anl0_hrxn, sum_Hrxn
import os
import yaml
from itertools import compress

# error
from errors import KeyErrorMessage


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

    def __init__(self, methods: list=[], force_generate_database:bool=False, force_generate_alternative_rxn:bool=False, dataframe_path:str=None):
        """
        ARGUMENTS
        ---------
        :methods:       [list] List of method names to use for calculation
                            of HoF. If empty, use all available methods.
                            (default=[])

        :force_generate_database:           [bool] (default=False)
                        Force the generation of the self.energies dataframe
                        from the folder containing individual species data
                        in yaml files.

        :force_generate_alternative_rxn:    [bool] (default=False) 
                        Force the generation of an alternative reaction
                        YAML file. Otherwise, use the existing one.
        
        :dataframe_path:                    [str] (default=None)
                        Load the self.energies dataframe from the pickle file
                        containing a previously saved self.energies dataframe.
        """

        # Load the methods to use
        with open('data/methods_keys.yaml', 'r') as f:
            self.methods_keys_dict = yaml.safe_load(f)

        # TODO: Check whether database related files (database, method_keys, alternative_CBH) exist or force download

        if len(methods)==0:
            # use all available methods in methods_keys dictionary
            self.methods = list(self.methods_keys_dict.keys())
            self.methods_keys = []
            for m in self.methods_keys_dict:
                self.methods_keys.extend(self.methods_keys_dict[m])
                self.methods_keys = list(set(self.methods_keys))
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
        
        # Load rankings
        self.rankings = load_rankings('data/rankings.yaml')
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
        
        self.rankings_rev = {}
        for k,l in self.rankings.items():
            for v in l:
                self.rankings_rev[v] = k
        
        # Generate Database
        if force_generate_database:
            self.energies = pd.DataFrame(generate_database('data/molecule_data')[0])[self.methods_keys+['source', 'DfH', 'DrxnH']]
        else:
            if dataframe_path:
                self.energies = pd.read_pickle(dataframe_path)
            else:
                constants = ['R', 'kB', 'h', 'c', 'amu', 'GHz_to_Hz', 'invcm_to_invm', 'P_ref', 'hartree_to_kcalpermole', 'hartree_to_kJpermole', 'kcalpermole_to_kJpermole','alias']
                # self.energies = pd.read_pickle('../autoCBH/main/data/energies_Franklin.pkl').drop(constants,axis=1) # for testing
                # self.energies = pd.read_pickle('./data/energies_Franklin.pkl').drop(constants,axis=1)
                self.energies = pd.read_pickle('./data/energies_Franklin_nan.pkl')
                # self.energies = generate_database('data/molecule_data/')[0] # something is different

                self.energies[['DrxnH']] = 0 # add delta heat of reaction column --> assume 0 for ATcT values
                # sort by num C then by SMILES length
                max_C = max([i.count('C') for i in self.energies.index])
                self.energies.sort_index(key=lambda x: x.str.count('C')*max_C+x.str.len(),inplace=True)
                ###
                # This stuff should go into test cases
                # self.energies.drop('CC(F)(F)F',axis=0, inplace=True) # for testing
                # self.energies.loc['CC(C)(F)F', ['avqz','av5z','zpe','ci_DK','ci_NREL','core_0_tz','core_X_tz','core_0_qz','core_X_qz',
                # 'ccT','ccQ','zpe_harm','zpe_anharm','b2plypd3_zpe','b2plypd3_E0','f12a','f12b','m062x_zpe',
                # 'm062x_E0','m062x_dnlpo','wb97xd_zpe','wb97xd_E0','wb97xd_dnlpo']] = nan

        # TODO: Alternative rxns
        # 1. check whether species exist
        # 2. if precurors don't exist, error message and delete reaction
        # 3. if exists then make it take priority over other possible reactions
        # 4. still compute other cbh rungs, and say in error messages that there are potential other reactions that are better
        #   - if better rung
        #   - if number of total species in reaction is less in another rung
        # I want options
        #   - force use alternative_rxn no matter what
        #   - let software decide (based on better rung (or less reactants?) if same then weight)

        if not os.path.isfile('data/alternative_rxn.yaml') or force_generate_alternative_rxn:
            generate_alternative_rxn_file('data/molecule_data', 'alternative_rxn')

        with open('data/alternative_rxn.yaml', 'r') as f:
            self.alternative_rxn = yaml.safe_load(f)
        
        self.error_messages = {}

        
    def generate_CBH_coeffs(self, species_list: list, saturate: int=1, allow_overshoot=False, include_target=True) -> list:
        """
        Generate a list of Pandas DataFrame objects that hold the coefficients 
        for every precursor created for CBH schemes of each target species.

        ARGUMENTS
        ---------
        :self:              [calcCBH] object

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

        :include_target:    [bool] (default=True) Include the target species with
                                a coefficient of -1

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
        for species in species_list:
            cbh = CBH.buildCBH(species, saturate, allow_overshoot=allow_overshoot) # generate CBH scheme
            # add to dictionary / lists
            all_rcts[species] = cbh.cbh_rcts
            all_pdts[species] = cbh.cbh_pdts
            highest_rung_per_molec.append(cbh.highest_cbh)
        
        # Find the highest CBH rung of all the CBH schemes
        highest_rung = max(highest_rung_per_molec)

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


    def calc_Hf(self, saturate:list=[1,9], max_rung:int=None, alt_rxn_options:str=None):
        """
        Calculate the heats of formation of species that do not have reference 
        values using the highest possible CBH scheme with the best possible level 
        of theory. This is done for the entire database.
        
        ARGUMENTS
        ---------
        :self:
        :saturate:      [list(int) or list(str)] List of integers representing the
                            atomic numbers of the elements to saturate precursor 
                            species. String representations of the elements 
                            also works. (default=[1,9] for hydrogen and fluorine)

        :max_rung:      [int] (default=None) Max CBH rung allowed for Hf calculation

        :alt_rxn_options: [str] (default=None)
                    - "ignore" (ignr)
                        Ignore alternative reactions that may exist and only use CBH 
                        priority system. (same result as None)
                    
                    - "best_alt" (best)
                        Always use the alternative reaction for all species that have
                        them even if better ranking CBH rungs exist. Just the best 
                        rung.

                    - "avg_alt" (avg)
                        Average all alternative reactions of a species and does not
                        use any default CBH rungs.

                    - "priority" (prty)
                        Include alternative reactions as possible rungs (USER DEFINED), 
                        but it will yield to better ranking CBH rungs if they exist. 
                        Further, they will weighted if equal rank to other reactions 
                        or a given CBH rung.

                    - "priority_precursor" (pprty)
                        Included alternative reactions and will follow CBH priority 
                        pipeline. However, it will prioritize the equations with the 
                        least number of precusors in the reaction since this metric
                        is often correlated to accuracy of the CBH equation.

        RETURN
        ------
        :self.energies:  [pd.DataFrame] columns for heat of formation, heat of 
                            reaction, and source.
        """
        
        alt_rxn_options_list = ['ignore', 'best_alt','avg_alt', 'priority', 'priority_precursor']
        if alt_rxn_options:
            if type(alt_rxn_options)!= str:
                raise TypeError('Arg "alt_rxn_options" must either be NoneType or str. If str, the options are: "ignore", "best_alt", "avg_alt", "priority", "priority_precursor".')
            elif alt_rxn_options not in alt_rxn_options_list:
                raise NameError('The available options for arg "alt_rxn_options" are: "ignore", "best_alt", "avg_alt", "priority", "priority_precursor".')

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

        Hf = {}
        Hrxn = {}
        # sort by the max distance between two atoms in a molecule
        simple_sort = lambda x: (max(max(CBH.mol2graph(AddHs(MolFromSmiles(x))).shortest_paths())))
        # sorted list of molecules that don't have any reference values
        sorted_species = sorted(self.energies[self.energies['source'].isna()].index.values, key=simple_sort)

        # cycle through molecules from smallest to largest
        for s in sorted_species:
            self.error_messages[s] = []

            if s in self.alternative_rxn and alt_rxn_options and alt_rxn_options != "ignore":
                ##### best alternative rxn option #####
                if alt_rxn_options == "best_alt":
                    best_alt = max(list(self.alternative_rxn[s].keys()))
                    alt_rxn = {}
                    alt_rxn[s] = self.alternative_rxn[s][best_alt]
                    alt_rxn[s][s] = -1

                    # check alternative rxn precursors existance in database
                    species = list(alt_rxn[s].keys())
                    if set(species).issubset(self.energies.index.values):
                        species_null = self.energies.loc[species, self.methods_keys].isna().all(axis=1)
                        # check ∆Hf
                        if True in self.energies.loc[species, 'source'].isna():
                            m_species = [species[i] for i, cond in enumerate(self.energies.loc[species, 'source'].isna().values) if cond==True]
                            missing_precursors = self._missing_precursor_str(m_species)
                            self.error_messages[s].append(f'The following precursors did not have usable reference heats of formation for the provided alternative reaction: {missing_precursors}')

                        # check QM energies
                        elif True in species_null:
                            missing_precursors = self._missing_precursor_str(species_null.index[species_null.values])
                            self.error_messages[s].append(f'The following precursors did not have usable QM values for the provided alternative reaction: {missing_precursors}')

                        else:
                            label = f'alt_rxn-{best_alt}-best'
                            Hrxn[s], Hf[s] = self.Hf(s, alt_rxn, skip_precursor_check=True)
                            final_Hrxn, final_Hf, label = self.choose_best_method(Hrxn[s], Hf[s], label)
                            self.energies.loc[s, 'DfH'] = final_Hf
                            self.energies.loc[s, 'DrxnH'] = final_Hrxn
                            self.energies.loc[s, 'source'] = label
                            
                            if len(self.error_messages[s]) == 0:
                                del self.error_messages[s]
                            continue

                    else:
                        m_species = [sp for sp in species if sp not in self.energies.index.values and sp != s]
                        missing_precursors = self._missing_precursor_str(m_species)
                        self.error_messages[s].append(f'Precursors in provided alternative reaction did not exist in database: {missing_precursors}')

                ##### avg alternative rxns option #####
                elif alt_rxn_options == "avg_alt":
                    pass
                ##### prioritize #####
                elif alt_rxn_options == "priority":
                    pass
                ##### prioritize precursor #####
                elif alt_rxn_options == "priority_precursor":
                    pass

            cbhs = []
            cbhs_rungs = []
            for i, sat in enumerate(saturate_nums):
                cbhs.append(CBH.buildCBH(s, sat, allow_overshoot=True))
                if max_rung is not None:
                    rung = max_rung if max_rung <= cbhs[-1].highest_cbh else cbhs[-1].highest_cbh
                else:
                    rung = cbhs[-1].highest_cbh
                cbhs_rungs.append(self.check_rung_usability(s, rung, cbhs[-1].cbh_rcts, cbhs[-1].cbh_pdts, saturate_syms[i]))

            if len(self.error_messages[s]) == 0:
                del self.error_messages[s]
            
            idx = np.where(np.array(cbhs_rungs) == np.array(cbhs_rungs).max())[0].tolist()
            cbhs = [cbhs[i] for i in idx]
            cbhs_rungs = [cbhs_rungs[i] for i in idx]
            s_syms = [saturate_syms[i] for i in idx]
            
            if len(cbhs_rungs) == 1:
                label = f'CBH-{str(cbhs_rungs[0])}-{s_syms[0]}'
            elif len(cbhs_rungs) > 1:
                label = f'CBH-{str(cbhs_rungs[0])}-avg('
                for i, sym in enumerate(s_syms):
                    if i > 0:
                        label += '+'
                    label += sym
                    if i == len(s_syms)-1:
                        label += ')'

            # compute Hrxn and Hf 
            if len(cbhs) == 1:
                # if only one saturation
                rxn = self.cbh_to_rxn(s, cbhs[0], cbhs_rungs[0])
                Hrxn[s], Hf[s] = self.Hf(s, rxn, skip_precursor_check=True)
            else:
                # otherwise weight the saturations
                Hrxn_ls = []
                Hf_ls = []
                for i, cbh in enumerate(cbhs):
                    rxn = self.cbh_to_rxn(s, cbh, cbhs_rungs[i])
                    Hrxn_s, Hf_s = self.Hf(s, rxn, skip_precursor_check=True)
                    Hrxn_ls.append(Hrxn_s)
                    Hf_ls.append(Hf_s)

                weights = {}
                for k in Hrxn_ls[0].keys():
                    weights[k] = self._weight(*[Hrxn_s[k] for Hrxn_s in Hrxn_ls])
                # weights = {method_keys : [weights_cbh_type]}
                Hrxn[s] = {}
                Hf[s] = {}
                for k in weights.keys():
                    Hrxn[s][k] = sum([weights[k][i]*Hrxn_s[k] for i, Hrxn_s in enumerate(Hrxn_ls)])
                    Hf[s][k] = sum([weights[k][i]*Hf_s[k] for i, Hf_s in enumerate(Hf_ls)])

            # Choose the best possible method to assign to the energies dataframe
            final_Hrxn, final_Hf, label = self.choose_best_method(Hrxn[s], Hf[s], label)
            self.energies.loc[s, 'DfH'] = final_Hf
            self.energies.loc[s, 'DrxnH'] = final_Hrxn
            self.energies.loc[s, 'source'] = label

        if len(self.error_messages.keys()) != 0:
            print(f'Process completed with errors in {len(self.error_messages.keys())} species')
            print(f'These errors are likely to have propagated for the calculation of heats of formation of larger species.')
            print(f'Updating the reference values of species in the database will improve accuracy of heats of formation.')
            print(f'To inspect the errors, run the calcCBH.print_errors() method.')

        return self.energies[['DfH', 'DrxnH', 'source']]


    def cbh_to_rxn(self, s: str, cbh: CBH.buildCBH, cbh_rung: int) -> dict:
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
        :s:         [str] SMILES string of the target species.
        
        :rxn:       [dict] Includes the negated reactant coefficient
                        and product coefficients. The species, s, is in
                        the dictionary with a negated coefficient.

        :skip_precursor_check:  [bool] (default=False)
                        Whether skip the procedure to check if precursors 
                        exist in database or if they have real values. 
                        Suggested to skip only when the cbh_rung is 
                        validated after check_rung_usability method is
                        used.

        RETURNS
        -------
        (Hrxn, Hf)  [tuple]
            :Hrxn:  [dict] Heat of reaction calculated for different levels
                        of theory using CBH.
                        {'ref' : val, *method : val}
            :Hf:    [dict] Heat of formation calculated for each of the same 
                        levels of theory in Hrxn. 
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
        coeff_arr = pd.DataFrame(rxn).T.sort_index(axis=1).values * -1 # products are negative, and reactants are positive
        nrg_arr = self.energies.loc[list(rxn[s].keys()), nrg_cols].sort_index().values
        matrix_mult = np.matmul(coeff_arr, nrg_arr)
        delE = {nrg_cols[i]:matrix_mult[0][i] for i in range(len(nrg_cols))}
        # Subtract off DfH from the energies column in case it is not 0
        # we must assert that the DfH is not computed for the target species
        delE['DfH'] = delE['DfH'] - self.energies.loc[s, 'DfH']
        
        Hrxn = {'ref':delE['DfH']}
        Hrxn = {**Hrxn, **dict(zip(self.methods, np.full(len(self.methods), nan)))}

        # TODO: How to choose what functions to include
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
        
        # Hf
        Hf = {k:v - Hrxn['ref'] for k,v in Hrxn.items()}

        return Hrxn, Hf


    def calc_Hf_alt_rxn(self, alt_rxn_options:str):
        """
        Calculate the heat of formation for an alternative reaction.
        """
        
        return


    def calc_Hf_allrungs(self, s: str, saturate: int or str=1) -> tuple:
        """
        Calculates the Hf of a given species at each CBH rung.
        Assumes that calc_Hf has already been run or the database
        has all of the necessary energies of molecules.

        ARGUMENTS
        ---------
        :s:         [str] SMILES string of the target species.
        :saturate:  [int or str] Atomic number of saturation element.

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
        s_cbh = CBH.buildCBH(s, saturate)

        Hf = {}
        Hrxn = {}
        for rung in range(s_cbh.highest_cbh):
            try:
                # TODO: error_messages will be sus 
                prepend_str = 'Computing all heats of formation at all rungs. \nErrors below are reflected by this operation'
                if s not in self.error_messages:
                    self.error_messages[s] = []
                self.error_messages[s].append(prepend_str)

                test_rung = self.check_rung_usability(s, rung, s_cbh.cbh_rcts, s_cbh.cbh_pdts, saturate_sym)
                if self.error_messages[s][-1] == prepend_str:
                    if len(self.error_messages[s])==1:
                        del self.error_messages[s]
                    else:
                        self.error_messages[s].pop() # delete prepend_str
                else:
                    self.error_messages[s].append('Conclude computation of all heats of formation for all rungs.')
                rxn = self.cbh_to_rxn(s, s_cbh, rung)
                Hrxn[rung], Hf[rung] = self.Hf(s, rxn, skip_precursor_check=True)
            except TypeError:
                print(f'Cannot compute CBH-{rung}')
                pass

        return Hrxn, Hf


    def choose_best_method(self, Hrxn: dict, Hf: dict, label: str):
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
            return nan, nan, 'NaN'

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
                        ### TODO: can get super long !!! ###
                        new_label = '' + label
                        for i, t in enumerate(theories):
                            if i == 0:
                                new_label += '//'+t
                            else:
                                new_label += '+'+t

                        return weighted_hrxn, weighted_hf, new_label

        return nan, nan, 'NaN'


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
        
        # if Hrxn is 0 for any species, return one-hot encoding
        if 0 in list(Hrxn):
            Hrxn = np.array(list(Hrxn))
            ind = np.where(Hrxn==0)[0]

            if len(ind) == 1:
                weights = np.zeros((len(Hrxn)))
                weights[ind[0]] = 1
                return weights.tolist()

            elif len(ind) > 1:
                weights = np.zeros((len(Hrxn)))
                weights[ind] = 1 / len(ind)
                return weights.tolist()
        
        # more common case where Hrxn is not exactly 0
        denom = 0
        # calculate denom
        for h in Hrxn:
            denom += (1/abs(h))

        weights = []
        # calculate weights
        for h in Hrxn:
            weights.append(1/abs(h) / denom)

        return weights

    
    def check_rung_usability(self, s: str, test_rung: int, cbh_rcts: dict, cbh_pdts: dict, label: str): 
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

        # 1. check existance of all reactants in database
        #   a. get precursors in a rung
        #   b. get the sources of those precursors
        for rung in reversed(range(test_rung+1)):
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
                new_sources = [s.split('//')[1].split('+')[0] if 'CBH' in s or 'alt_rxn' in s and len(s.split('//'))>1 else s for s in sources]
                source_rank = list([self.rankings_rev[source] for source in new_sources])
            else: # if a species Hf hasn't been computed yet, its source will be type==float, so move down a rung
                # This will appropriately trigger for overshooting cases
                test_rung -= 1
                continue
            
            # reaction dictionary
            rxn = add_dicts(cbh_pdts[rung], {k : -v for k, v in cbh_rcts[rung].items()}) 

            # 3. If the ranks are homoegenous, decompose precursors until all are of better rank than the target
            if len(set(source_rank))==1 and max(source_rank) >= rank and max(source_rank) != 1:
                verbose_error = False
                while max(source_rank) >= rank:
                    # if the worst source_rank is experimental we can break
                    if max(source_rank) == 1:
                        break
                    elif max(source_rank) > rank:
                        # This leaves potential to get better numbers for the precursors with larger ranks.
                        err_precur = list(compress(all_precursors, [r > rank for r in source_rank]))
                        missing_precursors = self._missing_precursor_str(err_precur)
                        self.error_messages[s].append(f"A precursor of the target molecule was computed with a level of theory that is worse than the target. \nConsider recomputing for: {missing_precursors}")

                    # This is where we need to decompose precursors until max(set_rank_source) < rank
                    #   Continuosly decomposing precurors that used averages of H+F will combinatorially increase the # of combinations...
                    #       In this case, might make sense to find worst rank in both cases, and choose the better one
                    #       If it's a tie, then go with the current saturation strategy
                    #           Or actually, maybe we should go with the current saturation all the time if it's an avg containing the saturation atom
                    # Need to think about whether I also need to decompose lower CBH rungs
                    #   I don't think I need to since if a higher rung is good, then it's the best option (unless # of reactants is < at lower rungs...)
                    #       Could try decomposing lower rungs, too if this is triggered, compare recorded number of reactants to that of lower rungs. 
                    #       If the number of reactants at the lower rungs is less, it's more favorable.
                    rxn_sources = self.energies.loc[list(rxn.keys()), 'source'].values.tolist() # sources of precursors in reaction

                    named_sources = [s.split('//')[1].split('+')[0] if 'CBH' in s or 'alt_rxn' in s and len(s.split('//'))>1 else s for s in rxn_sources]
                    # indices where theory is worse than rank
                    worse_idxs = np.where(np.array([self.rankings_rev[s] for s in named_sources]) >= rank)[0]
                    decompose_precurs = [list(rxn.keys())[i] for i in worse_idxs] # precursors where the theory is worse than rank and must be decomposed
                    # dictionaries holding a precursor's respective rung rung or saturation atom
                    d_rung = {precur : int(rxn_sources[i].split('//')[0].split('-')[1]) if len(rxn_sources[i].split('//')) > 1 else 'alt' for i, precur in enumerate(list(rxn.keys()))}
                    d_sat = {precur : rxn_sources[i].split('//')[0].split('-')[2].replace('avg(', '').replace(')', '').split('+') 
                                if len(rxn_sources[i].split('//')) > 1 and 'CBH' in rxn_sources[i] else 'alt' for i, precur in enumerate(list(rxn.keys()))}

                    for precur in decompose_precurs:
                        
                        # deciding how to choose how to saturate
                        if len(d_sat[precur]) == 1 and d_sat[precur] != 'alt':
                            p_sat = d_sat[precur][0]
                        elif label in d_sat[precur]:
                            p_sat = label
                        else:
                            # TODO: how to choose complicated precursors
                            p_sat = 'H'

                        if d_sat[precur] != 'alt':

                            p = CBH.buildCBH(precur, p_sat, allow_overshoot=True)
                            try: 
                                self.energies.loc[list(p.cbh_pdts[d_rung[precur]].keys()) + list(p.cbh_rcts[d_rung[precur]].keys())]
                                # To the original equation, add the precursor's precursors (rcts + pdts) and delete the precursor from the orig_rxn
                                rxn = add_dicts(rxn, {k : rxn[precur]*v for k, v in p.cbh_pdts[d_rung[precur]].items()}, {k : -rxn[precur]*v for k, v in p.cbh_rcts[d_rung[precur]].items()})
                                del rxn[precur]
                            except KeyError:
                                # new precursor does not exist in database
                                self.error_messages[s].append(f"Error occurred during decomposition of CBH-{test_rung}-{label} when checking for lower rung equivalency. Some reactants of {precur} did not exist in the database.")
                                self.error_messages[s][-1] += f"\nThere is a possibility that CBH-{test_rung}-{label} of {s} is equivalent to a lower rung, but this cannot be rigorously tested automatically."
                                verbose_error = True
                                break
                        else:
                            if set(self.alternative_rxn[precur][d_rung[precur]].keys()).issubset(self.energies.index.values):
                                rxn = add_dicts(rxn, self.alternative_rxn[precur][d_rung[precur]])
                                del rxn[precur]
                            else:
                                self.error_messages[s].append(f"Error occurred during decomposition of CBH-{test_rung}-{label} when checking for lower rung equivalency. Some reactants of {precur} did not exist in the database.")
                                self.error_messages[s][-1] += f"\nThere is a possibility that CBH-{test_rung}-{label} of {s} is equivalent to a lower rung, but this cannot be rigorously tested automatically."
                                verbose_error = True
                                break
                    
                    else: # if the for loop doesn't break (most cases)
                        rxn_sources = self.energies.loc[list(rxn.keys()), 'source'].values.tolist()
                        
                        if False not in [type(s)==str for s in rxn_sources]:
                            check_new_sources = [s.split('//')[1].split('+')[0] if 'CBH' in s or 'alt_rxn' in s and len(s.split('//'))>1 else s for s in rxn_sources]
                        else:
                            print('Uh Oh')
                            # Uh Oh
                            pass

                        source_rank = list([self.rankings_rev[s] for s in check_new_sources])
                        continue # continue while loop
                    
                    # will only trigger if the for-loop broke
                    break
                
                # TODO: Might make more sense to move this up a loop to be more rigorous
                #       It will slow down computation, but more accurate
                #       But also note that ^ might be unecessary since we simplify to the previous level of theory
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
        :s:             [str] SMILES string of the target species
        :top_rung:      [int] Rung that is currently being used for CBH of 
                            target. The function will check rungs lower than 
                            this (not including).

        :cbh_rcts:      [dict] Reactant precursors of the target molecule 
                            for each CBH rung.
                            {CBH_rung : {reactant_precursor : coeff}}
        :cbh_pdts:      [dict] Product precursors of the target molecule 
                            for each CBH rung.
                            {CBH_rung : {product_precursor : coeff}}
        :decomposed_rxn: [dict] A decomposed CBH precursor dictionary
                            {reactant/product_precursor : coeff}
        :verbose_error:  [bool] (default=False) Show equivalency dictionaries
                            in self.error_messages

        RETURNS
        -------
        (rung_equivalent, rung)    
            :rung_equivalent:      [bool] False if no matches, True if 
                                    equivalent to lower rung
            :rung:                 [int] Rung at given or equivalent rung
        """
        if verbose_error:
            self.error_messages[s][-1] += "\n\tReaction dictionaries hold the target species' precursors and respective coefficients where negatives imply reactants and positives are products."
            self.error_messages[s][-1] += f"\n\tBelow is the partially decomposed reaction: \n\t{precur_rxn}"
            self.error_messages[s][-1] += f"\n\tThe decomposed reaction will be subtracted from each CBH rung. For equivalency, all coefficients must be 0 which would yield an emtpy dictionary."

        for r in reversed(range(top_rung)): # excludes top_rung
            new_cbh_pdts = {k : -v for k, v in cbh_pdts[r].items()}
            total_prec = add_dicts(new_cbh_pdts, cbh_rcts[r], precur_rxn)
            if verbose_error:
                self.error_messages[s][-1] += f"\n\tSubtracted from CBH-{r}: {total_prec}"
            
            if not total_prec:
                # empty dict will return False
                return True, r

        return False, top_rung


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



def flatten(ls:list):
    """
    Flattens lists of lists when there are strings present in the outer list.
    From:
    https://stackoverflow.com/questions/5286541/

    ARGUMENTS
    ---------
    :ls:    [list]

    RETURNS
    -------
    [generator] flattened list as a generator obj
    """
    for l in ls:
        if hasattr(l, '__iter__') and not isinstance(l, str):
            for m in flatten(l):
                yield m
        else:
            yield l


if __name__ == '__main__':
    c = calcCBH(['CC(F)(F)(F)', 'CC(F)(F)C', 'CC(F)(F)C(F)(F)(F)', 'C(F)(F)(F)C(F)(F)C(F)(F)(F)',
    'CC(F)(F)C(F)(F)C', 'CC(F)(F)C(F)(F)C(F)(F)(F)', 'C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'])
    c.calc_Hf()

