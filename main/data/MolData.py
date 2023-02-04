from rdkit import Chem
from numpy import NaN
import yaml
import os
import sys
sys.path.append('.')
import pandas as pd
from numpy import nan, isnan

pd.set_option('display.float_format', '{:.6f}'.format)

class Molecule:
    def __init__(self, smiles:str, **kwargs):
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

        
def read_data(file: str, check_alternative_rxn=True):
    """
    Reads a yaml file containing data for a molecule.
    Data should be structured:
    - mol: RDKit Mol
    - smiles: SMILES str
    - alias: List(str)
    - alternative_rxn: dict - {CBH rung : {smiles : coeff}}
    - heat_of_formation: 
        - HoF_theory : value
    - heat_of_reaction:
        -HoF_theory: value
    - theory: 
        - method1:
            - method_key1 : energy
            - method_key2 : energy
        - method2
        .
        .
        .
    
    ARGUMENTS
    ---------
    :file:                      [str] file path to yaml file
    :check_alternative_rxn:     [bool] (default=True) Rigorous
                                check for correct SMILES in provided
                                alternative_rxn

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
    if check_alternative_rxn:
        record_bad_rungs = []
        new_alt_rxn_dict = {}
        if 'alternative_rxn' in molecule:
            for rung, a_cbh in molecule['alternative_rxn'].items():
                new_alt_rxn_dict[rung] = {}
                if type(rung) == int or type(rung) == float:
                    for smiles, coeff in a_cbh.items():
                        assert(type(smiles)==str)
                        try:
                            canon_smiles = Chem.CanonSmiles(smiles)
                        except:
                            print(f'ParseError: entered SMILES string is not valid')
                            print(f'\tRung {rung} in file: {file}')
                            print('\tThis reaction will not be added to file until fixed.')
                            record_bad_rungs.append(rung)
                            break
                        
                        if canon_smiles in new_alt_rxn_dict[rung]:
                            # triggers if equivalent smiles strings are in the alternative_rxn file
                            print(f'Error: alterantive_CBH for rung {rung} contains multiple smiles equivalent keys.\n\tFile found here: {file}')
                            print('\tThis reaction will not be added to file until fixed.')
                            record_bad_rungs.append(rung)
                            break
                        else:
                            new_alt_rxn_dict[rung][Chem.CanonSmiles(canon_smiles)] = coeff

                    else: # continue if inner loop didn't break
                        continue
                    # break if inner loop broke
                    break

                # if rung is not int or float
                else:
                    print(f'ArgumentError: entered "rung" number must be of type int or float. \n\tInstead got: {rung, type(rung)}.')
                    print(f'\tThis reaction will not be added to molecule until fixed.')
                    record_bad_rungs.append(rung)
                    break

            else: # if inner loop didn't break, add alterantive rxn
                molecule['alternative_rxn'][rung] = new_alt_rxn_dict[rung]
            
            for rung in record_bad_rungs:
                del molecule['alternative_rxn'][rung]
            if not molecule['alternative_rxn']: # if empty
                del molecule['alternative_rxn']

        
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
    :alternative_rxn: [dict] holds any alternative reactions for a given species
                        to possibly override the one derived from the CBH scheme. 
                        {target_smiles : {CBH_rung : {precursor_smiles : coeff}}}
    """
    energies = {}           # {smiles : {energy_type : value}}
    method_keys = {}        # {method : [necessary keys for calculation]}

    # load rankings
    with open(ranking_path, 'r') as f:
        rankings = yaml.safe_load(f)

    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if os.path.isfile(f):
            m = read_data(f, check_alternative_rxn=False)

            rank = nan
            energies[m['smiles']] = {}

            if 'theory' in m:
                if m['theory'] is not None:
                    for method in m['theory']:
                        # merge dicts to get expected structure
                        energies[m['smiles']] = {**energies[m['smiles']], **m['theory'][method]}
                        if method not in method_keys:
                            # does not take into account any typos or accidentally added keys
                            method_keys[method] = []
                        method_keys[method].extend(list(m['theory'][method]))
                        method_keys[method] = list(set(method_keys[method]))
                else:
                    print(f'File for the molecule: {m["smiles"]}, does not contain corresponding method and necessary values to the key "theory".')
                    print('This molecule will be skipped and not added to the database.')
                    del energies[m['smiles']]
                    continue
            else:
                print(f'File for the molecule: {m["smiles"]}, does not contain the key "theory" with corresponding method and necessary values.')
                print('This molecule will be skipped and not added to the database.')
                del energies[m['smiles']]
                continue
            
            if 'heat_of_formation' in m:
                for method in m['heat_of_formation']:
                    lvl_theory = method.split('//')[1].split('+')[0] if 'CBH' in method else method
                    if isnan(rank) or (not isnan(rank) and rankings[lvl_theory] < rank):
                        energies[m['smiles']]['DfH'] = m['heat_of_formation'][method]
                        energies[m['smiles']]['source'] = method
                        rank = rankings[lvl_theory] if 'CBH' in method else rankings[method]

            if 'heat_of_reaction' in m and 'heat_of_formation' in m:
                    energies[m['smiles']]['DrxnH'] = m['heat_of_reaction'][method]
            else:
                energies[m['smiles']]['DrxnH'] = 0

            if isnan(rank):
                energies[m['smiles']]['DfH'] = 0.
                energies[m['smiles']]['source'] = nan

    energies = pd.DataFrame(energies).T
    # energies[['DrxnH']] = 0
    max_C = max([i.count('C') for i in energies.index])
    energies.sort_index(key=lambda x: x.str.count('C')*max_C+x.str.len(),inplace=True)
    energies.to_dict()

    return energies, method_keys


def generate_alternative_rxn_file(folder_path:str, save_file:str=None):
    """
    Generates a YAML file containing alternative reactions.
    Structured:

    Target SMILES:
        - CBH rank [int/float]
            - precursor1 SMILES : coeff (negative for reactant, positive for product)
            - precursor2 SMILES : coeff
            .
            .
            .

    """

    alternative_rxn = {}    # {target smiles : {CBH_rung : {precursor smiles : coeff}}}
    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if os.path.isfile(f):
            m = read_data(f, check_alternative_rxn=True)
            if 'alternative_rxn' in m:
                alternative_rxn[m['smiles']] = m['alternative_rxn']
    
    if save_file:
        with open('./data/'+save_file+'.yaml','w') as f:
            yaml.dump(alternative_rxn, f, default_flow_style=False)
    
    return alternative_rxn


def add_alternative_rxns_to_database(alternative_rxn_file:str, database_folder:str):
    """
    Add reactions from alternative_rxn_file to their respective molecule files
    in the given folder.
    THIS WILL OVERWRITE ANY EXISTING REACTIONS IN THE MOLECULE FILE.

    ARGUMENTS
    ---------
    :alternative_rxn_file:  [str] path to the alternative_rxn_file
    :database_folder:       [str] path to the database folder

    RETURNS
    -------
    None
    """

    with open('data/alternative_rxn.yaml', 'r') as f:
        alternative_rxns = yaml.safe_load(f)
    for alt in alternative_rxns:
        alt = Chem.CanonSmiles(alt)
        mol_file = os.path.join(database_folder, alt+'.yaml')
        if os.path.isfile(mol_file):
            with open(mol_file, 'r') as f:
                mol = yaml.safe_load(f)
            if False in [isinstance(rank, (int, float)) for rank in alternative_rxns[alt]]:
                raise TypeError('CBH rank must be a number')
            for rank in alternative_rxns[alt]:
                if 'alternative_rxn' not in mol:
                    mol['alternative_rxn'] = {}
                # overwrite any existing reactions
                mol['alternative_rxn'][rank] = {Chem.CanonSmiles(smiles) : coeff for smiles, coeff in alternative_rxns[alt][rank].items()}
            
            with open(mol_file,'w') as f:
                yaml.dump(mol, f, default_flow_style=False)
        else:
            raise FileNotFoundError(f'File for molecule {alt} not found in database folder. File name must be RDKit canonical SMILES of molecule.')
            

def load_rankings(file=''):
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
            m = read_data(f)

            if 'alias' in m:
                if type(m['alias']) == str:
                    aliases[m['alias']] = m['smiles']
                elif type(m['alias']) == list and len(m['alias']) > 0:
                    for a in m['alias']:
                        aliases[a] = m['smiles']
                
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