from rdkit import Chem
from numpy import NaN
from easydict import EasyDict
import yaml
import os
import sys
sys.path.append('.')
import pandas as pd
from numpy import nan, isnan

class Molecule:
    def __init__(self, smiles, **kwargs):
        """
        Want kwargs to be of form:
        {method name: {values needed to compute energy : value}}

        Ranking:
        https://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/kinetics.html
        """

        # molecular descriptors
        self.mol = Chem.MolFromSmiles(smiles)
        self.smiles = Chem.MolToSmiles(self.mol)
        self.alias = []

        self.theory = {}
    
    def save(self):
        """
        Saves file with filename = self.smiles.
        Saves as yaml file with attributes of class.
        """
        return

        
def read_data(file: str):
    """
    Reads a yaml file containing data for a molecule.
    Data should be structured:
    - mol: RDKit Mol
    - smiles: SMILES str
    - alias: List(str)
    - alternative_CBH: dict - {CBH rung : {smiles : coeff}}
    - theory: 
        - method1:
            - energies: dict - {key : value}
            - heat_of_formation: float
        - method2
        .
        .
        .
    
    ARGUMENTS
    ---------
    :file:      [str] file path to yaml file

    RETURNS
    -------
    :molecule:  [dict] dictionary representation of yaml file
    
    """
    with open(file,'r') as f:
        molecule = yaml.safe_load(f)

    if 'smiles' in molecule.keys():
        if type(molecule['smiles']) == str:
            molecule['smiles'] = Chem.CanonSmiles(molecule['smiles'])
        else:
            raise TypeError('molecule["smiles"] must be a SMILES string.')
    else:
        raise KeyError('The molecule datafile must contain a SMILES string with key "smiles".')

    # Ensure canon SMILES are used for alternative CBH scheme equations
    if 'alternative_CBH' in molecule.keys():
        for rung, a_cbh in molecule['alternative_CBH'].items():
            if type(rung) == int:
                for species in a_cbh.keys():
                    assert(type(species)==str)
                    # seems slow
                    if Chem.CanonSmiles(species) != species:
                        # rename user-entered SMILES to canon SMILES
                        a_cbh[Chem.CanonSmiles(species)] = a_cbh[species]
                        del a_cbh[species]
    
    return molecule


def generate_database(folder_path: str, ranking_path: str = 'data/rankings.yaml'):
    """
    Read each molecule's data file and load into a pandas dataframe.
    Also create another dataframe or dictionary that contains each 
    theory's necessary keys for calculation.

    ARGUMENTS
    ---------
    :folder_path:     [str] path to folder holding molecular data

    RETURNS
    -------
    :energies:        [dict] holds energies dictionary to be converted to DataFrame
    :method_keys:     [dict] holds the keys necessary to compute the single point 
                        energy for each method
                        {method : [necessary keys fo calculation]}
    :alternative_CBH: [dict] holds any alternative reactions for a given species
                        to possibly override the one derived from the CBH scheme. 
                        {target_smiles : {CBH_rung : {precursor_smiles : coeff}}}
    """
    energies = {}           # {smiles : {energy_type : value}}
    method_keys = {}        # {method : [necessary keys for calculation]}
    alternative_CBH = {}    # {target smiles : {CBH_rung : {precursor smiles : coeff}}}

    # load rankings
    with open(ranking_path, 'r') as f:
        rankings = yaml.safe_load(f)

    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if os.path.isfile(f):
            m = EasyDict(read_data(f))

            rank = nan
            energies[m.smiles] = {}
            for method in m.theory.keys():
                # merge dicts to get expected structure
                if 'energies' in m.theory[method]:
                    energies[m.smiles] = {**energies[m.smiles], **m.theory[method].energies}
                    if method not in method_keys.keys():
                        # does not take into account any typos or accidentally added keys
                        method_keys[method] = [] # initialize
                    method_keys[method].extend(list(m.theory[method].energies))
                    method_keys[method] = list(set(method_keys[method]))
                
                hof_cond = 'heat_of_formation' in m.theory[method]
                if (isnan(rank) and hof_cond) or (not isnan(rank) and rankings[method] < rank and hof_cond):
                    energies[m.smiles]['DfH'] = m.theory[method].heat_of_formation
                    energies[m.smiles]['source'] = method
                    rank = rankings[method]

                if isnan(rank):
                    energies[m.smiles]['DfH'] = 0.
                    energies[m.smiles]['source'] = nan
            
            # Add alternative reactions to dictionary
            if 'alternative_CBH' in m:
                alternative_CBH[m.smiles] = m.alternative_CBH
    
    energies = pd.DataFrame(energies).T
    energies.loc[:,energies.columns != 'source'] = energies.loc[:,energies.columns != 'source'].fillna(0)
    energies.to_dict()

    return energies, method_keys, alternative_CBH


def load_rankings(file: str=''):
    """
    Load Ranking file for different levels of theory.
    Reverse the file such that it returns a dictionary of form
    {rank : list(methods)}
    
    Default is in 'data/rankings.yml'

    ARGUMENTS
    ---------
    :file:      [str] file path to yaml file holding rankings

    RETURNS
    -------
    :rankings:  [dict] ranking of each level of theory.
                    {# : [theory1, theory2, ...]}
    """
    with open(file, 'r') as f:
        rankings = yaml.safe_load(f)

    rephrase = {}
    for k,v in rankings.items():
        if v not in list(rephrase.keys()):
            rephrase[v] = [k] # list
        elif v in list(rephrase.keys()):
            rephrase[v].append(k)

    return rephrase


def generate_alias_file(folder_path: str):
    """
    Generate a file that contains the various aliases
    of molecules.

    TODO: warn and prevent overwriting unless forced

    ARGUMENTS
    ---------
    """
    aliases = {}
    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if os.path.isfile(f):
            m = EasyDict(read_data(f))

            if 'alias' in m:
                if type(m.alias) == str:
                    aliases[m.alias] = m.smiles
                elif type(m.alias) == list and len(m.alias) > 0:
                    for a in m.alias:
                        aliases[a] = m.smiles
                
    # TODO: save to a file
    with open('alias.yaml', 'w') as f:
        yaml.dump(aliases, f)

    return aliases


def search_alias(name: str, alias_filepath='data/alias.yaml'):
    """
    Searches through a file containing alternative names for
    molecules. Returns the Canon SMILES of a given alias.

    ARGUMENTS
    ---------
    :name:          [str] Alternative name to search
    :alias_file:    [str] YAML filepath of alias file to searc

    RETURNS
    -------
    :Canon_SMILES:  [str] Canon SMILES that matches the alias file
    """
    with open(alias_filepath, 'r') as f:
        alias = yaml.safe_load(f)
    if name in alias:
        return alias[name]
    else:
        KeyError(f'Given "name" not in alias file.')