import CBH
from CBH import add_dicts
import numpy as np
import pandas as pd
from numpy import nan, isnan
from rdkit.Chem import MolFromSmiles, MolToSmiles, AddHs, CanonSmiles, GetPeriodicTable
from data.molData import load_rankings
from hrxnMethods import anl0_hrxn, sum_Hrxn
import os
import yaml


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

    def __init__(self, methods: list=[], force_generate_database=False):
        """
        ARGUMENTS
        ---------
        :methods:       [list] List of method names to use for calculation
                            of HoF. If empty, use all available methods.
                            (default=[])
        """

        # Load the methods to use
        with open('data/methods_keys.yaml', 'r') as f:
            self.methods_keys_dict = yaml.safe_load(f)

        # TODO: Check whether database related files (database, method_keys, alternative_CBH) exist or force download

        if len(methods)==0:
            # use all available methods in methods_keys dictionary
            self.methods = list(self.methods_keys_dict.keys())
            self.methods_keys = []
            for m in self.methods_keys_dict.keys():
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

        # with open('data/alternative_CBH.yaml', 'r') as f:
        #     self.alternative_CBH = yaml.safe_load(f)
        
        # TODO: create energies DataFrame using only self.method_keys

        constants = ['R', 'kB', 'h', 'c', 'amu', 'GHz_to_Hz', 'invcm_to_invm', 'P_ref', 'hartree_to_kcalpermole', 'hartree_to_kJpermole', 'kcalpermole_to_kJpermole','alias']
        # self.energies = pd.read_pickle('../autoCBH/main/data/energies_Franklin.pkl').drop(constants,axis=1) # for testing
        self.energies = pd.read_pickle('./data/energies_Franklin.pkl').drop(constants,axis=1)
        self.energies[['DrxnH']] = 0 # add delta heat of reaction column --> assume 0 for ATcT values
        # sort by num C then by SMILES length
        self.energies.sort_index(key=lambda x: x.str.count('C')*max(x.str.count('C'))+x.str.len(),inplace=True)
        # self.energies.drop('CC(F)(F)F',axis=0, inplace=True) # for testing
        # self.energies.loc['CC(C)(F)F', ['avqz','av5z','zpe','ci_DK','ci_NREL','core_0_tz','core_X_tz','core_0_qz','core_X_qz',
        # 'ccT','ccQ','zpe_harm','zpe_anharm','b2plypd3_zpe','b2plypd3_E0','f12a','f12b','m062x_zpe',
        # 'm062x_E0','m062x_dnlpo','wb97xd_zpe','wb97xd_E0','wb97xd_dnlpo']] = nan
        
        self.error_messages = {}

        # self.Hrxn = self.calc_Hrxn()
        
    def generate_CBHs(self, species_list: list, saturate: int=1) -> list:
        """
        Generate a list of Pandas DataFrame objects that hold the coefficients 
        for every precursor created for CBH schemes of each target species.

        ARGUMENTS
        ---------
        :self:          [calcCBH] object
        :species_list:  [list] List of SMILES strings with target molecules.
        :saturate:      [int or str] Integer representing the atomic number 
                            of the element to saturate residual species with. 
                            String representation of the element also works.
                            (default=1)

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
        all_targets = [] # list of SMILES strings representing the target molecules
        for species in species_list:
            cbh = CBH.buildCBH(species, saturate) # generate CBH scheme
            # add to dictionary / lists
            all_rcts[cbh.smiles] = cbh.cbh_rcts
            all_pdts[cbh.smiles] = cbh.cbh_pdts
            all_targets.append(cbh.smiles) # add standardized SMILES string representations
        
        # Find the highest CBH rung of all the CBH schemes
        highest_rung_per_molec = [max(pdt.keys()) for pdt in all_pdts.values()]
        highest_rung = max(highest_rung_per_molec)

        dfs = [] # initialize DataFrame list
        # Cycle through each CBH rung
        for rung in range(highest_rung):
            df = {}
            # Cycle through each species in species_list
            for species in all_targets:
                if rung <= max(all_pdts[species].keys()):
                    df[species] = all_pdts[species][rung]
                    df[species].update((precursor, coeff * -1) for precursor,coeff in all_rcts[species][rung].items())
                    df[species].update({species:-1})
            dfs.append(pd.DataFrame(df).fillna(0).T)
        return dfs


    def calc_Hf(self, saturate:list=[1,9]):
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

        RETURN
        ------
        :self.energies:  [pd.DataFrame] columns for heat of formation, heat of 
                            reaction, and source.
        """
        
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

        # cycle through sorted molecules
        for s in sorted_species:
            cbhs = []
            cbhs_rungs = []
            self.error_messages[s] = []
            for i, sat in enumerate(saturate_nums):
                cbhs.append(CBH.buildCBH(s, sat, allow_overshoot=True))
                cbhs_rungs.append(self.check_rung_usability(s, cbhs[-1].highest_cbh, cbhs[-1].cbh_rcts, cbhs[-1].cbh_pdts, saturate_syms[i], sat))

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
                Hrxn[s], Hf[s] = self.Hf(s, cbhs[0], cbhs_rungs[0])
            else:
                # otherwise weight the saturations
                Hrxn_ls = []
                Hf_ls = []
                for i, cbh in enumerate(cbhs):
                    Hrxn_s, Hf_s = self.Hf(s, cbh, cbhs_rungs[i])
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

        return self.energies[['DfH', 'DrxnH', 'uncertainty', 'source']]

    def Hf(self, s: str, cbh: CBH.buildCBH, cbh_rung: int) -> tuple:
        """
        Helper method to calculate the heat of formation (and reaction) given 
        a species and it's cbh scheme (CBH.buildCBH object).

        ARGUMENTS
        ---------
        :s:         [str] SMILES string of the target species.
        :cbh:       [CBH.buildCBH obj] CBH scheme of the target species.
        :cbh_rung:  [int] CBH rung number to calculate the HoF

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

        hartree_to_kJpermole = 2625.499748 # (kJ/mol) / Hartree

        rxn = {} 
        # add rxn of a target molecule's highest possible CBH level
        rxn[s] = cbh.cbh_pdts[cbh_rung] # products
        rxn[s].update((p,coeff*-1) for p,coeff in cbh.cbh_rcts[cbh_rung].items()) # negative reactants
        rxn[s].update({s:-1}) # target species

        # check if there are any precursors that don't exist in the database
        precursors_not_tabulated = [p for p in rxn[s] if p not in self.energies.index.values]
        if len(precursors_not_tabulated) != 0:
            print('Missing the following species in the database:')
            for pnt in precursors_not_tabulated:
                print(f'\t{pnt}')
            return # break
        
        # TODO: Fix this since it will break the caclulation and user can't control order
        for precursor in rxn[s]:
            if self.energies.loc[precursor,'source'] ==  'NaN':
                print(f'Must restart with this molecule first: {precursor}')
                return 
        
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
                if m == 'anl0':
                    if 0 not in self.energies.loc[rxn[s].keys(), ['avqz', 'av5z', 'zpe']].values:
                        Hrxn['anl0'] = anl0_hrxn(delE)
                elif 0 not in self.energies.loc[rxn[s].keys(), self.methods_keys_dict[m]].values:
                    Hrxn[m] = sum_Hrxn(delE, *self.methods_keys_dict[m])
        
        # Hf
        Hf = {k:v - Hrxn['ref'] for k,v in Hrxn.items()}
        # print(s, Hf)

        return Hrxn, Hf


    def calc_Hf_allrungs(self, s: str, saturate: int=1) -> tuple:
        """
        Calculates the Hf of a given species at each CBH rung.
        Assumes that calc_Hf has already been run or the database
        has all of the necessary energies of molecules.

        ARGUMENTS
        ---------
        :s:         [str] SMILES string of the target species.
        :saturate:  [int] Atomic number of saturation element.

        RETURN
        ------
        (Hrxn, Hf): [tuple]
            Hrxn    [dict] {rung : heat of reaction}
            Hf      [dict] {rung : heat of formation}
        """

        s = CanonSmiles(s) # standardize SMILES
        s_cbh = CBH.buildCBH(s, saturate)

        Hf = {}
        Hrxn = {}
        for rung in range(s_cbh.highest_cbh):
            try:
                Hrxn[rung], Hf[rung] = self.Hf(s, s_cbh, rung)
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
                        return Hrxn[theory], Hf[theory], label+'_'+theory
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
                        return hrxns[0], hfs[0], label + '_' + theories[0]
                    
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
                                new_label += '_'+t
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

    
    def check_rung_usability(self, s: str, test_rung: int, cbh_rcts: dict, cbh_pdts: dict, label: str, saturate:int=1): 
        """
        Method that checks a given CBH rung's usability by looking for missing 
        reactants or whether all precursors were derived using CBH schemes of 
        the same level of theory. Errors will cause the method to move down to
        the next rung and repeat the process. It stores any errors to the 
        calcCBH.error_messages attribute which can be displayed with the 
        calcCBH.print_errors() method. 

        ??? Still need to check if there are homogeneous level of theory 
        values in database for the calculation ???

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
        
        nrg_cols = ['avqz','av5z','zpe','ci_DK','ci_NREL','core_0_tz','core_X_tz','core_0_qz','core_X_qz',
        'ccT','ccQ','zpe_harm','zpe_anharm','b2plypd3_zpe','b2plypd3_E0','f12a','f12b','m062x_zpe',
        'm062x_E0','m062x_dnlpo','wb97xd_zpe','wb97xd_E0','wb97xd_dnlpo']

        for rung in reversed(range(test_rung+1)):
            # check existance of all reactants in database
            all_precursors = list(cbh_rcts[rung].keys()) + list(cbh_pdts[rung].keys())

            try:
                sources = self.energies.loc[all_precursors, 'source'].values.tolist()
                # if all values contributing to the energy of a precursor are NaN move down a rung
                species_null = self.energies.loc[all_precursors, nrg_cols].isnull().all(axis=1)
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
                else:
                    pass
            
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
                    self.error_messages[s][-1] += f'\n\tRung will move down to {test_rung}.'
                continue

            # Try checking precursors' reactants
            ### Unfinished
            # This would get tricky since I would need to check source for each of these species too
            # total_precursors = add_dicts(cbh_rcts[test_rung], cbh_pdts[test_rung])
            # for precursor in total_precursors.keys():
            #     p_cbh = CBH.buildCBH(precursor, saturate)
            #     p_rcts = add_dicts(p_cbh.cbh_pdts, p_cbh.cbh_rcts)

            
            # check sources of all reactants
            ####
            # TODO
            # Need to change so that if the source of the target species is worse than those of the reactants its okay
            # ex. if all reactants are f12, but target will use dnlpo, that is totally fine
            # TODO
            # String parsing won't work with new ranking system

            ###
            # Maybe...
            # Compute the Hf for rung below and if equal, move down a rung...?
            #   Expensive compute... but mathmatically correct
            ####

            # sources will be np.nan if not computed yet
            if False not in [type(s)==str for s in sources]:
                sources = [s.split('_')[0] for s in sources]
                set_source = list(set(sources))
            else:
                # if a species Hf hasn't been computed yet, it's source will be type==float, so move down a rung
                test_rung -= 1
                continue

            if len(set_source) == 1: # homogenous sources
                if set_source[0] != 'ATcT': 
                    self.error_messages[s].append(f'All precursor species for CBH-{test_rung}-{label} had the same reference level of theory. \n\tThis implies that this rung is equivalent to CBH-{test_rung-1}-{label}. \n\tUsing CBH-{test_rung-1}-{label} instead.')
                    test_rung -= 1
                    continue
                else: # okay to use this rung if they are experimental values
                    return test_rung
            else: # heterogenous sources are good
                return test_rung
        return test_rung

    
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


if __name__ == '__main__':
    c = calcCBH(['CC(F)(F)(F)', 'CC(F)(F)C', 'CC(F)(F)C(F)(F)(F)', 'C(F)(F)(F)C(F)(F)C(F)(F)(F)',
    'CC(F)(F)C(F)(F)C', 'CC(F)(F)C(F)(F)C(F)(F)(F)', 'C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'])
    c.calc_Hf()

