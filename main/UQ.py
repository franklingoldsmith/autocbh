import numpy as np
import pandas as pd
from calcCBH import calcCBH
import CBH
import sys
sys.path.append('.')
from rdkit.Chem import MolFromSmiles, AddHs
from copy import copy
from tqdm import tqdm

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

    :priority_by_coeff: [bool] (default=False)
        Choose to prioritize CBH rungs based on the total number of reactants in 
        the reaction. If True, the CBH rung with the least number of reactants 
        will be used. If False, the highest rung number will by prioritized. 
        If alt_rxn_option is "priority", then it will also take alternative 
        reactions into account.

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
                    priority_by_coeff:bool=False, 
                    max_rung:int=None, 
                    alt_rxn_option:str=None, 
                    force_generate_database:bool=False, 
                    force_generate_alternative_rxn:bool=False):
        
        self.num_simulations = num_simulations
        self.saturate = saturate
        self.priority_by_coeff = priority_by_coeff
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
        """
        self.simulation_results = np.zeros((len(self.calcCBH.energies.index.values), self.num_simulations + 1)) # extra for the mean

        self.calcCBH.calc_Hf(self.saturate, self.priority_by_coeff, self.max_rung, self.alt_rxn_option)
        self.simulation_results[:, 0] = self.calcCBH.energies.loc[:, 'DfH'].values

        self.simulation_results[self.non_nan_species_ind, 1:] = copy(self.init_simulation_matrix)

        # sort criteria
        simple_sort = lambda x: (max(max(CBH.mol2graph(AddHs(MolFromSmiles(x))).shortest_paths())))

        # sorted list of molecules that don't have any reference values
        sorted_species = sorted(self.calcCBH.energies[self.calcCBH.energies['uncertainty'].isna()].index.values, key=simple_sort)

        # cycle through molecules from smallest to largest
        for s in tqdm(sorted_species):
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

        self.calcCBH.calc_Hf(self.saturate, self.priority_by_coeff, self.max_rung, self.alt_rxn_option)
        self.simulation_results[0,:] = self.calcCBH.energies.loc[:, 'DfH'].values

        # sort criteria
        simple_sort = lambda x: (max(max(CBH.mol2graph(AddHs(MolFromSmiles(x))).shortest_paths())))

        for i in tqdm(range(0, self.init_simulation_matrix.shape[1])):
            self.calcCBH.energies.loc[self.non_nan_species, 'DfH'] = self.init_simulation_matrix[:,i]
            # sorted list of molecules that don't have any reference values
            sorted_species = sorted(self.calcCBH.energies[self.calcCBH.energies['uncertainty'].isna()].index.values, key=simple_sort)

            # cycle through molecules from smallest to largest
            for s in sorted_species:
                self.calcCBH.calc_Hf_from_source(s, inplace=True)

            self.simulation_results[i+1,:] = self.calcCBH.energies.loc[:, 'DfH'].values
        
        self.simulation_results = self.simulation_results.T
