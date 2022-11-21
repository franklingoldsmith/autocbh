from rdkit import Chem
from numpy import NaN
from easydict import EasyDict
import yaml

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

        
def read_data(file: str):
    """
    Reads a yaml file containing data for a molecule.
    Data should be structured:
    - mol: RDKit Mol
    - smiles: SMILES str
    - alias: List(str)
    - theory: 
        - method1:
            - energies: dict
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

    if 'smile' in molecule:
        if type(molecule['smile']) == str:
            molecule['smile'] = Chem.CanonSmiles(molecule['smile'])
        else:
            raise TypeError('molecule["smile"] must be a SMILES string.')
    else:
        raise KeyError('The molecule datafile must contain a SMILES string with key "smile".')
    
    return molecule


def load_rankings(file: str=''):
    """
    Load Ranking file for different levels of theory.
    
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