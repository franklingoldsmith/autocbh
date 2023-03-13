import numpy as np
from rdkit import Chem
import CBH
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import colors
import sys
sys.path.append('.')


class TN:

    def __init__(self, species:str or list, max_rung:int=100, saturate:int or str=1, surface_smiles:str=None, **kwargs):
        """
        Class to build and visualize thermochemical network.
        In other words, see what molecules depend on one another using the CBH method.

        ARGUMENTS
        ---------
        :species:       [str or list(str)]
                    SMILES string or list of SMILES strings that contains the target species 
                    for which to generate a thermochemical network.
        
        :max_rung:      [int] (default=100)
                    Maximum CBH rung that you want for the largest target species.
                    If max_rung is greater than the highest possible CBH rung, that
                    CBH rung will be used.

        :saturate:      [int or str] (default=1 or 'H')
                    The int or str representation of the default molecule that will 
                    saturate the heavy atoms. Usually 'H' (1), but is also often 
                    'F' (9) or 'Cl' (17). Currently it only supports halogens. 

         :surface_smiles:    [str] (default=None)
                    Valid SMILES string representing the surface atom that the given 
                    molecule is adsorbed to or physiosorbed to. Must be a single atom.
                    i.e., '[Pt]'
                    NOT INCLUDING THIS FOR ADSORBATES WILL RETURN MISLEADING TNs.
        """
        
        # check species argument
        if isinstance(species, str):
            try:
                self.species = [Chem.CanonSmiles(species)]
            except:
                raise NameError(f'Arg "species" must be a valid SMILES string. Instead, {species} was given.')
        elif isinstance(species, list):
            try:
                self.species = [Chem.CanonSmiles(smiles) for smiles in species]
            except:
                raise NameError(f'Arg "species" must contain valid SMILES strings. Check input.')
        else:
            raise TypeError(f'Arg "species" must either be a valid SMILES string or a list containing valid SMILES strings. Instead {type(species)} was given.')

        # check max_rung argument
        if not isinstance(max_rung, int):
            raise TypeError(f'Arg "max_rung" must be an integer. Instead {type(max_rung)} was given.')
        elif max_rung < 0:
            raise NameError(f'Arg max_rung must be a non-negative number. Instead {max_rung} was given.')
        else:
            self.max_rung = max_rung
        
        # check saturate argument
        ptable = Chem.GetPeriodicTable()
        if not isinstance(saturate, (int, str)):
            raise TypeError(f'Arg "saturate" must be an integer or string representing an element to saturate fragements. Instead, {type(saturate)} was given.')
        elif isinstance(saturate, int):
            try:
                ptable.GetElementSymbol(saturate)
                self.saturate = saturate
            except:
                raise NameError(f'Arg "saturate" must be a integer that maps to an element in the periodic table. Instead, {saturate} was given.')
        elif isinstance(saturate, str):
            try:
                self.saturate = ptable.GetAtomicNumber(saturate)
            except:
                raise NameError(f'Arg "saturate" must be a string that maps to an element in the periodic table. Instead, {saturate} was given.')
            

        self.surface_smiles = surface_smiles
        
        self.cbhs = [CBH.buildCBH(smiles, saturate=saturate, allow_overshoot=True, surface_smiles=surface_smiles) for smiles in self.species]
        self.highest_rungs = [cbh.highest_cbh if cbh.highest_cbh <= self.max_rung else self.max_rung for cbh in self.cbhs]
        
        all_smiles = [list(cbh.cbh_rcts[rung].keys()) + list(cbh.cbh_pdts[rung].keys()) for i, cbh in enumerate(self.cbhs) for rung in range(self.highest_rungs[i])]

        self.all_smiles = list(set([smiles for sublist in all_smiles for smiles in sublist] + self.species))

        self.index2smiles = {i : smiles for i, smiles in enumerate(self.all_smiles)}
        self.smiles2index = {v:k for k, v in self.index2smiles.items()}

        self.graph = nx.DiGraph()


    def build(self):
        """
        Build a graph representing the thermochemical network.

        ARGUMENTS
        ---------
        None

        RETURNS
        -------
        :self.graph:    [networkx DiGraph]
            nodes = {species_id : num_times_used}
            edges = {(higher_rung_species_id, lower_rung_species_id) : rung}
        """

        for i, smiles in enumerate(self.species):
            species_index = self.smiles2index[smiles]
            # add node for target
            if not self.graph.has_node(species_index):
                self.graph.add_node(species_index, weight=1)
            else:
                self.graph.nodes[species_index]['weight'] += 1

            self._build_recurs(smiles, self.highest_rungs[i])

        self.graph = nx.relabel_nodes(self.graph, self.index2smiles)
        return self.graph

    
    def _build_recurs(self, smiles:str, prev_cbh_rung:int):
        """
        Recursive helper function to add to graph based on precursor of best rung.

        ARGUMENTS
        ---------
        :smiles:    [str]
                Smiles of the current target molecule

        :prev_cbh_rung:  [int]
                CBH rung that was used to generate the target smiles
                as a precursor.
        
        RETURNS
        -------
        None
        """
        
        if prev_cbh_rung == 0:
            return 
        try:
            cbh = CBH.buildCBH(smiles, saturate=self.saturate, allow_overshoot=True, surface_smiles=self.surface_smiles)
        except KeyError:
            return

        highest_rung = cbh.highest_cbh if cbh.highest_cbh <= self.max_rung else self.max_rung

        species_index = self.smiles2index[smiles]
        for precursors_dict in [cbh.cbh_pdts[highest_rung], cbh.cbh_rcts[highest_rung]]:
            for s, coeff in precursors_dict.items():
                if s not in self.smiles2index.keys():
                    self.smiles2index[s] = max(self.smiles2index.values()) + 1
                    self.index2smiles[max(self.smiles2index.values())] = s
                    self.all_smiles += [s]

                precursor_index = self.smiles2index[s]

                if not self.graph.has_node(precursor_index):
                    self.graph.add_node(precursor_index, weight=coeff)
                else:
                    self.graph.nodes[precursor_index]['weight'] += coeff

                self.graph.add_edge(species_index, precursor_index, rung=highest_rung)
                                
                self._build_recurs(s, highest_rung)


    def visualize(self, figsize:tuple=(24,8), save_fig_path:str=None):
        """
        Visualize network as a tree (DAG). Edges are color-coded for CBH rung.
        Nodes are color-coded for the importance of a given molecule.

        ARGUMENTS
        ---------
        :figsize:       [tuple] (default=(24,8))
                The (width, height) of the pyplot figure.
        
        :save_fig_path: [str] (default=None)
                The local path to save figure to.

        RETURNS
        -------
        None
        """
        ax = plt.figure(figsize=figsize)
        pos = graphviz_layout(self.graph, prog='dot')
        edge_cmap = plt.cm.Accent
        node_cmap = plt.cm.Blues

        nx.draw(self.graph, pos, with_labels=False, arrows=True, node_size=0, font_size=9, edge_color=[self.graph[u][v]['rung'] for u,v in self.graph.edges], edge_cmap=edge_cmap)

        node_colors = [v if v < len(self.graph.nodes) else len(self.graph.nodes) for v in nx.get_node_attributes(self.graph, 'weight').values()]
        
        nx.draw_networkx_nodes(self.graph, pos=pos, node_size=0)
        
        labels = nx.draw_networkx_labels(self.graph, pos=pos, 
                                        labels={node:node for node in self.graph.nodes.keys()},
                                        bbox=dict(edgecolor='black', boxstyle='round,pad=0.5'))

        # node color
        color_scale = 0.2
        normalize_node_colors = (np.array(node_colors) - min(node_colors))/(max(node_colors) - min(node_colors))
        for t, c in zip(labels.values(), normalize_node_colors):
            #manipulate indiviual text objects
            t.set_backgroundcolor(node_cmap(c-color_scale))
        
        # plot node colorbar (weight)
        norm = colors.Normalize(min(node_colors)+color_scale*max(node_colors), max(node_colors)+color_scale*max(node_colors), clip=True)
        cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=node_cmap), pad=0.005)
        cbar.set_label(label=f'Usage Density (not shown {max(node_colors)}+)', size=20)
        cbar.ax.tick_params(labelsize=20)

        # plot edge colorbar (weight)
        edge_cmap_bounds = np.array(list(set(nx.get_edge_attributes(self.graph, 'rung').values())))
        edge_norm = colors.BoundaryNorm(edge_cmap_bounds, edge_cmap.N)
        cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap), pad=0.05)
        cbar.set_label(label='CBH Rung', size=20)
        cbar.ax.tick_params(labelsize=20)
        if save_fig_path:
            plt.savefig(save_fig_path)
        plt.show()