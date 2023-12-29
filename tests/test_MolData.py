#############################
# Test Suite for data.molData.py #
#############################

from autocbh.data.molData import load_rankings, generate_database, read_data
from pytest import raises
import sys
from rdkit.Chem import CanonSmiles
from numpy import nan, isnan
sys.path.append('.')


class TestReadData:

    def test_raise_error_if_file_not_valid(self):
        with raises(Exception):
            read_data('fakefile.yaml')
    
    def test_raise_error_if_file_is_not_yaml(self):
        with raises(Exception):
            read_data('tests/dummy_data/empty.pkl')

    def test_raise_error_if_smiles_not_keyword(self):
        with raises(Exception):
            read_data('tests/dummy_data/no_smiles.yaml')

    def test_raise_error_if_smiles_invalid(self):
        with raises(Exception):
            read_data('tests/dummy_data/bad_smiles.yaml')

    def test_ignores_bad_alternative_rxns(self):
        file = 'tests/dummy_data/dummy_database_bad/bad_alternative_rxns.yaml'
        molec = read_data(file)
        assert 'alternative_rxn' not in molec


class TestGenerateDatabase:
    folder = 'tests/dummy_data/dummy_database'
    rankings_file = 'tests/dummy_data/rankings.yaml'
    
    def test_raise_error_if_rankings_path_not_valid(self):
        with raises(Exception):
            generate_database(folder_path=self.folder, ranking_path='fakefile.yaml')

    def test_raise_error_if_folder_path_not_valid(self):
        with raises(Exception):
            generate_database(folder_path='fakefolder', ranking_path=self.rankings_file)

    def test_raise_error_if_folder_is_empty(self):
        """Should raise error if folder has no yaml files."""
        with raises(Exception):
            generate_database(folder_path='tests/dummy_data/invalid_folder', ranking_path=self.rankings_file)

    def test_skip_molecule_if_no_theory_keyword(self):
        # targets no_theory_keyord.yaml
        # SMILE in folder: O=C(Cl)NSF
        # Since there is no theory keyword, energies dataframe won't contain O=C(Cl)NSF
        energies, _ = generate_database(self.folder, self.rankings_file)
        assert 'O=C(Cl)NSF' not in energies.index.values

    def test_skip_molecule_if_theory_keyword_is_empty(self):
        # targets empty_theory_keyword.yaml
        # SMILE in folder: O=C(Cl)NSCl
        # Since there is nothing under the theory keyword, nergies dataframe won't contain O=C(Cl)NSF
        energies, _ = generate_database(self.folder, self.rankings_file)
        assert 'O=C(Cl)NSCl' not in energies.index.values

    def test_already_heat_of_formation_inputted_in_file(self):
        energies, _ = generate_database(self.folder, self.rankings_file)
        smiles = CanonSmiles('CC(F)(F)C(C)(F)F')
        Hf_info = energies.loc[smiles, ['DfH', 'DrxnH', 'source']].to_dict()
        assert Hf_info == {'DfH': -964.1640793168818, 'DrxnH': 0.0, 'source': 'CBHavg-(2-H, 2-F)//f12b'}

    def test_method_keys_is_correct(self):
        _, method_keys = generate_database(self.folder, self.rankings_file)
        for method, keys in method_keys.items():
            method_keys[method] = set(keys)
        assert method_keys == {'b2plypd3': set(['b2plypd3_E0']), 
                               'f12b': set(['b2plypd3_zpe', 'f12b']), 
                               'm062x': set(['m062x_E0']), 
                               'm062x_dlpno': set(['m062x_zpe', 'm062x_dlpno']), 
                               'wb97xd': set(['wb97xd_E0']), 
                               'wb97xd_dlpno': set(['wb97xd_dlpno', 'wb97xd_zpe'])}
        
    def test_energies_df_has_correct_DfH_values(self):
        energies, _ = generate_database(self.folder, self.rankings_file)
        smiles = CanonSmiles('CCCC')
        assert energies.loc[smiles, 'DfH'] == 0.

    def test_energies_df_has_correct_DrxnH_values(self):
        energies, _ = generate_database(self.folder, self.rankings_file)
        smiles = CanonSmiles('CCCC')
        assert energies.loc[smiles, 'DrxnH'] == 0.0

    def test_energies_df_has_correct_source(self):
        energies, _ = generate_database(self.folder, self.rankings_file)
        smiles = CanonSmiles('CCCC')
        assert isnan(energies.loc[smiles, 'source'])


# Can add tests for alternative reaction handling