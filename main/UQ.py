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

    def __init__(self, num_simulations, calcCBH_obj=None, 
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
            self.c = calcCBH_obj
        else:
            self.c = calcCBH(methods=methods, force_generate_database=force_generate_database, 
                        force_generate_alternative_rxn=force_generate_alternative_rxn, 
                        dataframe_path=dataframe_path, alternative_rxn_path=alternative_rxn_path)
        
        self.species = copy(self.c.energies.index.values)

        self.non_nan_species = self.c.energies.loc[~np.isnan(self.c.energies.loc[:, 'uncertainty'].values), 'uncertainty'].index
        uncs = self.c.energies.loc[self.non_nan_species,'uncertainty'].values

        means_matrix = np.tile(np.array(self.c.energies.loc[self.non_nan_species, 'DfH'].values), (num_simulations, 1))
        # ATcT is 2 sigma uncertainty
        sigma_matrix = np.tile(np.array(uncs)/2, (num_simulations, 1))

        self.init_simulation_matrix = np.random.normal(loc=means_matrix, scale=sigma_matrix)

        self.simulation_matrix_results = 0
    
    
    def run(self):
        """
        Run uncertainty quantification operation.

        ARGUMENTS
        ---------
        None

        RETURNS
        -------
        :simulation_matrix_results: [np.array] Heats of formation
                    shape = (num_species x num_simulations)
        """

        self.simulation_matrix_results = np.zeros((self.num_simulations + 1, len(self.c.energies.index.values))) # extra for the mean

        self.c.calc_Hf(self.saturate, self.priority_by_coeff, self.max_rung, self.alt_rxn_option)
        self.simulation_matrix_results[0,:] = self.c.energies.loc[:, 'DfH'].values

        # sort criteria
        simple_sort = lambda x: (max(max(CBH.mol2graph(AddHs(MolFromSmiles(x))).shortest_paths())))

        for i in tqdm(range(0, self.init_simulation_matrix.shape[0])):
            self.c.energies.loc[~np.isnan(self.c.energies.loc[:, 'uncertainty'].values), 'DfH'] = self.init_simulation_matrix[i,:]
            # sorted list of molecules that don't have any reference values
            sorted_species = sorted(self.c.energies[self.c.energies['uncertainty'].isna()].index.values, key=simple_sort)

            # cycle through molecules from smallest to largest
            for s in sorted_species:
                self.c.calc_Hf_from_source(s, inplace=True)

            self.simulation_matrix_results[i+1,:] = self.c.energies.loc[:, 'DfH'].values
        
        self.simulation_matrix_results = self.simulation_matrix_results.T

        return self.simulation_matrix_results
