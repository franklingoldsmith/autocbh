import numpy as np
from calcCBH import calcCBH
import CBH
import os, sys
sys.path.append('.')
from rdkit.Chem import MolFromSmiles, AddHs
from copy import copy
from tqdm import tqdm
from itertools import product

class uncertainty_quantification:
    """
    Uncertainty quantification class for calcCBH objects. It computes the
    heat of formation of every species in the database for a user-defined
    number of simulations where each simulation is a initialized by 
    experimentally derived heats of formations randomly sampled from a 
    normal distribution using their provided uncertainties.

    ARGUMENTS
    ---------
    :num_simulations:   [int] (default=1000)
            Number of simulations / initial configurations

    :calcCBH_obj:       [calcCBH] (default=None)
            An already defined calcCBH object to use for UQ. If provided, 
            the following arguments are ignored.
    
    :methods:       [list] (default=[])
        List of method names to use for calculation of HoF. If empty, use 
        all available methods.

    :force_generate_database:   [bool] (default=False)
        Force the generation of the self.energies dataframe from the folder 
        containing individual species data in yaml files.

    :force_generate_alternative_rxn:    [bool] (default=False) 
        Force the generation of an alternative reaction YAML file. Otherwise, 
        use the existing one.
    
    :dataframe_path:    [str] (default=None)
        Load the self.energies dataframe from the pickle file containing a 
        previously saved self.energies dataframe.
    
    :saturate:  [list(int) or list(str)] (default=[1,9] for hydrogen and fluorine)
        List of integers representing the atomic numbers of the elements to 
        saturate precursor  species. String representations of the elements also 
        works. 

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
        "abs_coeff":    val = 7
        "rel_coeff":    val = 0
        "rung":         val = 1

    :max_rung:      [int] (default=None) 
        Max CBH rung allowed for Hf calculation

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

        - "include" (priority) - or maybe "include"?
            Included alternative reactions and will follow CBH priority 
            protocol. 
    
    ATTRIBUTES
    ----------
    :calcCBH:   [calcCBH obj]

    :species:   [list] List of SMILES of species in self.calcCBH.energies.

    :non_nan_species: [list] List of SMILES of species in self.calcCBH.energies
        with experimental values with provided uncertainty values.
    
    :non_nan_species_ind: [list] List of indices corresponding to self.non_nan_species

    :init_simulation_matrix: [np.array] shape=(num_non_nan_species, num_simulations)
        2D array containing the randomly sampled initial experimental values

    :simulation_results:    [np.array] shape=(num_species, num_simulations+1)
        2D array containing results of all simulations for all species.
        First column corresponds to the mean configuration.
    """
    def __init__(self, num_simulations=1000, calcCBH_obj:calcCBH=None, 
                    methods: list=[], 
                    dataframe_path:str=None, 
                    alternative_rxn_path:str=None, 
                    saturate:list=[1,9], 
                    priority:str="abs_coeff", 
                    max_rung:int=None, 
                    alt_rxn_option:str=None, 
                    force_generate_database:bool=False, 
                    force_generate_alternative_rxn:bool=False):
        
        self.methods = methods
        self.dataframe_path = dataframe_path
        self.alternative_rxn_path = alternative_rxn_path
        self.force_generate_database = force_generate_database
        self.force_generate_alternative_rxn = force_generate_alternative_rxn

        self.num_simulations = num_simulations
        self.saturate = saturate
        self.priority = priority
        self.max_rung = max_rung
        self.alt_rxn_option = alt_rxn_option
        
        if calcCBH_obj:
            self.calcCBH = calcCBH_obj
        else:
            self.calcCBH = calcCBH(methods=methods, force_generate_database=force_generate_database, 
                        force_generate_alternative_rxn=force_generate_alternative_rxn, 
                        dataframe_path=dataframe_path, alternative_rxn_path=alternative_rxn_path)
        
        self.species = copy(self.calcCBH.energies.index.values)

        self.non_nan_species = self.calcCBH.energies.loc[~np.isnan(self.calcCBH.energies.loc[:, 'uncertainty'].values), 'uncertainty'].index.values
        self.non_nan_species_ind = self.calcCBH.energies.index.get_indexer(self.non_nan_species)
        uncs = self.calcCBH.energies.loc[self.non_nan_species,'uncertainty'].values

        means_matrix = np.tile(np.array(self.calcCBH.energies.loc[self.non_nan_species, 'DfH'].values), (num_simulations, 1))
        # ATcT is 2 sigma uncertainty
        sigma_matrix = np.tile(np.array(uncs)/2, (num_simulations, 1))

        self.init_simulation_matrix = np.random.normal(loc=means_matrix, scale=sigma_matrix).T

        self.simulation_results = 0
    
    
    def run(self):
        """
        Run uncertainty quantification operation in a vectorized manner.
        Speed limiting step is the first step where the self.calcCBH.energies
        dataframe is filled out once using the provided means and the 'source'
        column is computed. Afterwards, the heats of formation are computed
        in a vectorized manner for each species that does not have experimental
        values with attributed uncertainties.

        ARGUMENTS
        ---------
        None

        RETURNS
        -------
        None - results found in self.simulation_results attribute

            :self.simulation_results: [np.array] Heats of formation
                        shape = (num_species x num_simulations)
        """
        self.simulation_results = np.zeros((len(self.calcCBH.energies.index.values), self.num_simulations + 1)) # extra for the mean

        self.calcCBH.calc_Hf(self.saturate, self.priority, self.max_rung, self.alt_rxn_option)
        self.simulation_results[:, 0] = self.calcCBH.energies.loc[:, 'DfH'].values

        self.simulation_results[self.non_nan_species_ind, 1:] = copy(self.init_simulation_matrix)

        # sorted list of molecules that don't have any reference values
        sorted_species = sorted(self.calcCBH.energies[self.calcCBH.energies['uncertainty'].isna()].index.values, key=self.simple_sort)

        # cycle through molecules from smallest to largest
        pbar = tqdm(sorted_species)
        for s in pbar:
            pbar.set_description(f'Number of Species')
            i_s = self.calcCBH.energies.index.get_loc(s)
            weighted_Hrxn, weighted_Hf = self.calcCBH.calc_Hf_from_source_vectorized(s, self.simulation_results[:, 1:], self.calcCBH.energies.index)

            self.simulation_results[i_s, 1:] = weighted_Hf


    def run_slow(self):
        """
        Run uncertainty quantification operation.

        ARGUMENTS
        ---------
        None

        RETURNS
        -------
        :simulation_results: [np.array] Heats of formation
                    shape = (num_species x num_simulations)
        """

        self.simulation_results = np.zeros((self.num_simulations + 1, len(self.calcCBH.energies.index.values))) # extra for the mean

        self.calcCBH.calc_Hf(self.saturate, self.priority, self.max_rung, self.alt_rxn_option)
        self.simulation_results[0,:] = self.calcCBH.energies.loc[:, 'DfH'].values

        pbar = tqdm(range(0, self.init_simulation_matrix.shape[1]))
        for i in pbar:
            pbar.set_description(f'Sample {i+1}')
            self.calcCBH.energies.loc[self.non_nan_species, 'DfH'] = self.init_simulation_matrix[:,i]
            # sorted list of molecules that don't have any reference values
            sorted_species = sorted(self.calcCBH.energies[self.calcCBH.energies['uncertainty'].isna()].index.values, key=self.simple_sort)

            # cycle through molecules from smallest to largest
            for s in sorted_species:
                self.calcCBH.calc_Hf_from_source(s, inplace=True)

            self.simulation_results[i+1,:] = self.calcCBH.energies.loc[:, 'DfH'].values
        
        self.simulation_results = self.simulation_results.T
    

    def run_cbh_selection(self, alt_rxn_option:list=None, priority:list=None):
        """
        UQ of different CBH selection options across randomly sampled initial 
        heats of formation for experimental species.

        ARGUMENTS
        ---------
        :alt_rxn_option:    [list] (default = None) 
                A list of strings specifying the type of options to include.
                Options: 'ignore', 'best_alt','avg_alt', 'include'
                None will automatically include all four options.
        
        :priority:          [list] (default = None)
                A list of strings specifying how to prioritize reaction schema.
                Options: 'abs_coeff', 'rel_coeff', 'rung'
                None will automatically include all three options.
                
        RETURNS
        -------
        None --> will alter self.simulation_results in-place

            :self.simulation_results: [np.array] Heats of formation
                shape = (alt_rxn_combinations, num_species, num_simulations)
        """

        if not isinstance(alt_rxn_option, (list, type(None))):
            raise TypeError(f'Arg "alt_rxn_option" must either be of type List or NoneType. Instead, {type(alt_rxn_option)} was given.')
        alt_rxn_option_list_check = ['ignore', 'best_alt','avg_alt', 'include']
        if alt_rxn_option is not None:
            for option in alt_rxn_option:
                if option:
                    if type(option)!= str:
                        raise TypeError('List items within "alt_rxn_option" must be a str. If str, the options are: "ignore", "best_alt", "avg_alt", "include".')
                    elif option not in alt_rxn_option_list_check:
                        raise NameError('The available options for items within "alt_rxn_option" are: "ignore", "best_alt", "avg_alt", "include".')

        if alt_rxn_option is None:
            alt_rxn_option_list = ['ignore', 'best_alt', 'avg_alt', 'include']
        else:
            alt_rxn_option_list = alt_rxn_option
        
        if not isinstance(priority, (list, type(None))):
            raise TypeError(f'Arg "priority" must either be of type List or NoneType. Instead, {type(priority)} was given.')
        priority_list = ["abs_coeff", "rel_coeff", "rung"]
        if priority is not None:
            for p in priority:
                if not isinstance(p, str):
                    raise TypeError(f'Arg "priority" must be a str. Instead, {type(p)} was given. The options are: "abs_coeff", "rel_coeff", "rung".')
                if p not in priority_list:
                    raise NameError(f'The available options for arg "priority" are: "abs_coeff", "rel_coeff", "rung". {p} was given instead.')
        else:
            priority = ["abs_coeff", "rel_coeff", "rung"]
        
        combos = list(product(alt_rxn_option_list, priority))

        sorted_species = sorted(self.calcCBH.energies[self.calcCBH.energies['uncertainty'].isna()].index.values, key=self.simple_sort)

        self.simulation_results = np.zeros((len(combos), len(self.calcCBH.energies.index.values), self.num_simulations + 1))

        self.simulation_results[:, self.non_nan_species_ind, 1:] = copy(self.init_simulation_matrix)
        pbar = tqdm(enumerate(combos))
        for c, combo in pbar:
            pbar.set_description(f'alt rxn option: {combo[0]} | priority: {combo[1]}')
            alt_option, p_option = combo
            with HiddenPrints():
                self.calcCBH.calc_Hf(self.saturate, priority=p_option, max_rung=self.max_rung, alt_rxn_option=alt_option)

            self.simulation_results[c, :, 0] = self.calcCBH.energies.loc[:, 'DfH'].values
            # cycle through molecules from smallest to largest
            for s in sorted_species:
                i_s = self.calcCBH.energies.index.get_loc(s)
                weighted_Hrxn, weighted_Hf = self.calcCBH.calc_Hf_from_source_vectorized(s, self.simulation_results[c, :, 1:], self.calcCBH.energies.index)

                self.simulation_results[c, i_s, 1:] = weighted_Hf
            self.calcCBH.energies.loc[self.calcCBH.energies['uncertainty'].isna(), 'source'] = np.nan

        return combos


    def run_cbh_sat(self, sat_list:list, alt_rxn_option:str='include', priority:str="abs_coeff"):
        """
        UQ for different saturation strategies. Automatically tries the values
        within the list individually and the average.
        """
        alt_rxn_option_list_check = ['ignore', 'best_alt','avg_alt', 'include']
        if alt_rxn_option:
            if type(alt_rxn_option)!= str:
                raise TypeError('Arg "alt_rxn_option" must be a str. The options are: "ignore", "best_alt", "avg_alt", "include".')
            elif alt_rxn_option not in alt_rxn_option_list_check:
                raise NameError('The available options for "alt_rxn_option" are: "ignore", "best_alt", "avg_alt", "include".')
        
        priority_list = ["abs_coeff", "rel_coeff", "rung"]
        if not isinstance(priority, str):
            raise TypeError(f'Arg "priority" must be a str. Instead, {type(priority)} was given. The options are: "abs_coeff", "rel_coeff", "rung".')
        if priority not in priority_list:
            raise NameError(f'The available options for arg "priority" are: "abs_coeff", "rel_coeff", "rung". {priority} was given instead.')

        if not isinstance(sat_list, list):
            raise TypeError(f'Arg "sat_list" type must be list. Instead, {type(sat_list)} was given.')

        sats = [[sat] for sat in sat_list]
        sats.append(sat_list)

        sorted_species = sorted(self.calcCBH.energies[self.calcCBH.energies['uncertainty'].isna()].index.values, key=self.simple_sort)

        self.simulation_results = np.zeros((len(sats), len(self.calcCBH.energies.index.values), self.num_simulations + 1))

        self.simulation_results[:, self.non_nan_species_ind, 1:] = copy(self.init_simulation_matrix)
        pbar = tqdm(enumerate(sats))
        for sat_i, sat in pbar:
            pbar.set_description(f'Saturation: {sat}')
            with HiddenPrints():
                self.calcCBH.calc_Hf(sat, priority=priority, max_rung=self.max_rung, alt_rxn_option=alt_rxn_option)

            self.simulation_results[sat_i, :, 0] = self.calcCBH.energies.loc[:, 'DfH'].values

            # cycle through molecules from smallest to largest
            for s in sorted_species:
                i_s = self.calcCBH.energies.index.get_loc(s)
                weighted_Hrxn, weighted_Hf = self.calcCBH.calc_Hf_from_source_vectorized(s, self.simulation_results[sat_i, :, 1:], self.calcCBH.energies.index)

                self.simulation_results[sat_i, i_s, 1:] = weighted_Hf
            
            self.calcCBH.energies.loc[self.calcCBH.energies['uncertainty'].isna(), 'source'] = np.nan

        return sats
    

    def simple_sort(self, x):
        """Sorting algorithm used for smallest to largest
        Computes the longest path across a molecule.
        The np.inf condition is used to avoid any issues with
        physiosorbed species."""
        arr = np.array(CBH.mol2graph(AddHs(MolFromSmiles(x))).shortest_paths())
        return max(arr[arr < np.inf])


class HiddenPrints:
    """
    Mute any print statements.
    https://stackoverflow.com/questions/8391411/
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout