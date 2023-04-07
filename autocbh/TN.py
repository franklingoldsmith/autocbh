import numpy as np
from rdkit import Chem
import CBH
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import colors
import sys
sys.path.append('.')
import os
import yaml
import pandas as pd
import calcCBH

class thermochemical_network:

    def __init__(self, species:str or list or calcCBH.calcCBH, max_rung:int=np.inf, saturate:int or str=1, surface_smiles:str=None):
        """
        Class to build and visualize thermochemical network.
        In other words, see what molecules depend on one another using the CBH method.

        ARGUMENTS
        ---------
        :species:       [str or list(str) or calcCBH.calcCBH]
                    SMILES string or list of SMILES strings that contains the target species 
                    for which to generate a thermochemical network.
        
        :max_rung:      [int] (default=np.inf)
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
        
        self.smiles2rung = {}
        self.smiles2sat = {}
        self.smiles2rank = {}
        self._uses_dataframe = False
        
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

        elif isinstance(species, calcCBH.calcCBH):
            if 'source' not in species.energies.columns:
                raise NameError('Arg "energies_df" DataFrame must have the method of ∆Hf computation in a column named "source".')
            self._uses_dataframe = True
            smiles2source = dict(zip(list(species.energies.index), species.energies.loc[:,'source'].values.tolist()))
            self.species = list(smiles2source.keys())

            for s in self.species:
                if len(smiles2source[s].split('//')) > 1:
                    # CBH used
                    if 'avg' in smiles2source[s].split('//')[0]:
                        # ex. CBHavg-(N-S, N-S, N-alt)
                        self.smiles2rung[s] = [float(sub.split('-')[0]) for sub in smiles2source[s].split('//')[0][8:-1].split(', ')]
                        self.smiles2sat[s] = [sub.split('-')[1] for sub in smiles2source[s].split('//')[0][8:-1].split(', ')]
                    else:
                        # ex. CBH-N-S
                        self.smiles2rung[s] = [float(smiles2source[s].split('//')[0].split('-')[1])]
                        self.smiles2sat[s] = [smiles2source[s].split('//')[0].split('-')[2]]
                    self.smiles2rank[s] = species.rankings_rev[smiles2source[s].split('//')[1].split('+')[0]]
                else:
                    # experimental
                    self.smiles2rung[s] = [None]
                    self.smiles2sat[s] = [None]
                    self.smiles2rank[s] = species.rankings_rev[smiles2source[s]]
        else:
            raise TypeError(f'Arg "species" must be a valid SMILES string, a list containing valid SMILES strings, or a DataFrame computed from calcCBH. Instead {type(species)} was given.')

        # check max_rung argument
        if not isinstance(max_rung, int) and max_rung != np.inf:
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
        if isinstance(species, (str, list)):
            cbhs = [CBH.buildCBH(smiles, saturate=saturate, allow_overshoot=True, surface_smiles=surface_smiles) for smiles in self.species]
            self.highest_rungs = [cbh.highest_cbh if cbh.highest_cbh <= self.max_rung else self.max_rung for cbh in cbhs]
        
            all_smiles = [list(cbh.cbh_rcts[rung].keys()) + list(cbh.cbh_pdts[rung].keys()) for i, cbh in enumerate(cbhs) for rung in range(self.highest_rungs[i])]

            self.all_smiles = list(set([smiles for sublist in all_smiles for smiles in sublist] + self.species))

        elif isinstance(species, calcCBH.calcCBH):
            self.highest_rungs = [1]*len(self.species) # doesn't matter since not used directly
            self.all_smiles = self.species

        self.index2smiles = {i : smiles for i, smiles in enumerate(self.all_smiles)}
        self.smiles2index = {v:k for k, v in self.index2smiles.items()}

        self.graph = nx.DiGraph()
        # Build graph
        self._build()


    def _build(self):
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
                self.graph.add_node(species_index)

            self._build_recurs(smiles, self.highest_rungs[i])

        # add num_ancestors attribute to nodes
        remove_nodes = []
        is_DAG = nx.is_directed_acyclic_graph(self.graph)
        for n in self.graph:
            if self.graph.in_degree(n) == 0 and self.graph.out_degree(n) == 0:
                remove_nodes.append(n)
            if is_DAG:
                self.graph.nodes[n]['num_ancestors'] = len(nx.ancestors(self.graph, n))

        # clean graph
        #   remove nodes without any edges
        self.graph.remove_nodes_from(remove_nodes)

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
        
        def _inner_loop(smiles, cbh, highest_rung):
            species_index = self.smiles2index[smiles]
            for precursors_dict in [cbh.cbh_pdts[highest_rung], cbh.cbh_rcts[highest_rung]]:
                for s, coeff in precursors_dict.items():
                    if s not in self.smiles2index.keys():
                        self.smiles2index[s] = max(self.smiles2index.values()) + 1
                        self.index2smiles[max(self.smiles2index.values())] = s
                        self.all_smiles += [s]

                    precursor_index = self.smiles2index[s]

                    if not self.graph.has_node(precursor_index):
                        self.graph.add_node(precursor_index)

                    if not self.graph.has_edge(species_index, precursor_index):
                        if self._uses_dataframe:
                            self.graph.add_edge(species_index, precursor_index, rung=highest_rung, rank=self.smiles2rank[s])
                        else:
                            self.graph.add_edge(species_index, precursor_index, rung=highest_rung)

                        self._build_recurs(s, highest_rung)


        if smiles in self.smiles2rung:            
            for rung, sat in zip(self.smiles2rung[smiles], self.smiles2sat[smiles]):
                if rung is None or sat is None:
                    return

                cbh = CBH.buildCBH(smiles, saturate=sat, allow_overshoot=True, surface_smiles=self.surface_smiles)
                highest_rung = rung if rung <= self.max_rung else self.max_rung

                _inner_loop(smiles, cbh, highest_rung)
        else:
            try:
                cbh = CBH.buildCBH(smiles, saturate=self.saturate, allow_overshoot=True, surface_smiles=self.surface_smiles)
            except KeyError:
                return

            highest_rung = cbh.highest_cbh if cbh.highest_cbh <= self.max_rung else self.max_rung

            _inner_loop(smiles, cbh, highest_rung)



    def visualize(self, graph:nx.DiGraph=None, relabel_node_mapping:dict or str=None, reverse_relabel_node_mapping:bool=None, 
                  figsize:tuple=(24,8), title:str=None, save_fig_path:str=None, dpi:int or float=None, label_font_size:int=12,
                  edge_rank:bool=False):
        """
        Visualize network as a tree (DAG). Edges are color-coded for CBH rung.
        Nodes are color-coded for the importance of a given molecule.

        ARGUMENTS
        ---------
        :graph: [nx.DiGraph or None] (default=None)
                If none, it uses the class attributed 'graph'. 
                Otherwise, the user may specify a specific graph to visualize.
                (typically a subgraph using thermochemical_network.descendent_subraph_of)
                A directed acyclic graph (DAG) is expected.

        
        :relabel_node_mapping:  [dict or str] (default=None)
                Dictionary that maps SMILES strings to an alternative name.
                Or a path to a YAML file containing the mapping dictionary.
                e.g.) {C(F)(F) : ch2cf2}
        
        :reverse_relabel_node_mapping: [bool] (default=None)
                Whether to reverse the provided dictionary in arg relabel_node_mapping.
                Will map values to keys rather than keys to values.
                
        :figsize:       [tuple] (default=(24,8))
                The (width, height) of the pyplot figure.
        
        :title:         [str] (default=None)
                Title to print on the plot.
        
        :save_fig_path: [str] (default=None)
                The local path to save figure to.
        
        :dpi:   [float or int] (default=None)
                The picture quality "dots per inch" to save figure.
                Will save image at this quality, but will not show up in
                the output of this method.

        :label_font_size:   [int] (default=12)
                Font size of node labels on the plot.

        RETURNS
        -------
        None
        """
        if not graph:
            graph = self.graph.copy()
        
        visualize(graph, 
                  relabel_node_mapping=relabel_node_mapping, 
                  reverse_relabel_node_mapping=reverse_relabel_node_mapping,
                  figsize=figsize,
                  title=title,
                  save_fig_path=save_fig_path,
                  dpi=dpi,
                  label_font_size=label_font_size, 
                  edge_rank=edge_rank)


    def descendent_subgraph_of(self, smiles:str):
        """
        Returns the subgraph that only includes descendents of the 
        specified species (including itself). 
        This is useful to pair with thermochemical_network.visualize.

        ARGUMENTS
        ---------
        :smiles:    [str]
            The SMILES string of the root node to generate the subgraph.

        RETURNS
        -------
        :subgraph:  [nx.DiGraph] 
            Subgraph containing the specified node and its descendents.
        """

        descendants = nx.descendants(self.graph, smiles)
        descendants.add(smiles)
        subgraph = self.graph.subgraph(descendants).copy()
        return subgraph
    

def visualize(graph:nx.DiGraph, relabel_node_mapping:dict or str=None, reverse_relabel_node_mapping:bool=None, 
                figsize:tuple=(24,8), title:str=None, save_fig_path:str=None, dpi:int or float=None, label_font_size:int=12,
                edge_rank:bool=False):
    """
    Visualize network as a tree (DAG). Edges are color-coded for CBH rung.
    Nodes are color-coded for the importance of a given molecule.

    ARGUMENTS
    ---------
    :graph: [nx.DiGraph or None] (default=None)
            If none, it uses the class attributed 'graph'. 
            Otherwise, the user may specify a specific graph to visualize.
            (typically a subgraph using thermochemical_network.descendent_subraph_of)
            A directed acyclic graph (DAG) is expected.

    
    :relabel_node_mapping:  [dict or str] (default=None)
            Dictionary that maps SMILES strings to an alternative name.
            Or a path to a YAML file containing the mapping dictionary.
            e.g.) {C(F)(F) : ch2cf2}
    
    :reverse_relabel_node_mapping: [bool] (default=None)
            Whether to reverse the provided dictionary in arg relabel_node_mapping.
            Will map values to keys rather than keys to values.
            
    :figsize:       [tuple] (default=(24,8))
            The (width, height) of the pyplot figure.
    
    :title:         [str] (default=None)
            Title to print on the plot.
    
    :save_fig_path: [str] (default=None)
            The local path to save figure to.
    
    :dpi:   [float or int] (default=None)
            The picture quality "dots per inch" to save figure.
            Will save image at this quality, but will not show up in
            the output of this method.

    :label_font_size:   [int] (default=12)
            Font size of node labels on the plot.

    RETURNS
    -------
    None
    """

    is_DAG = nx.is_directed_acyclic_graph(graph)
    for n in graph:
        if is_DAG:
            graph.nodes[n]['num_ancestors'] = len(nx.ancestors(graph, n))

    if relabel_node_mapping and isinstance(relabel_node_mapping, dict):
        if reverse_relabel_node_mapping:
                alias_rev = {v:k for k, v in relabel_node_mapping.items()}
                graph = nx.relabel_nodes(graph, alias_rev)
        else:
            graph = nx.relabel_nodes(graph, relabel_node_mapping)

    elif relabel_node_mapping and isinstance(relabel_node_mapping, str):
        if relabel_node_mapping[-5:] != '.yaml':
            print(f'Not a vaild YAML file (must end in .yaml). Please check your input to arg "relabel_node_mapping": f{relabel_node_mapping}. Continuing without relabeling.')
            graph = graph
        elif os.path.isfile(relabel_node_mapping):
            with open(relabel_node_mapping, 'r') as f:
                alias = yaml.safe_load(f)
            if reverse_relabel_node_mapping:
                alias_rev = {v:k for k, v in alias.items()}
                graph = nx.relabel_nodes(graph, alias_rev)
            else:
                graph = nx.relabel_nodes(graph, alias)


    # ax = plt.figure(figsize=figsize)
    fig, axs = plt.subplots(ncols=3,figsize=figsize, gridspec_kw={"width_ratios":[1, 0.01, 0.01]})
    pos = graphviz_layout(graph, prog='dot')
    edge_cmap = plt.cm.Dark2
    node_cmap = plt.cm.Blues
    if title:
        plt.suptitle(title, fontsize=30)
    
    used_rank = False
    if edge_rank:
        try:
            edge_color = [graph[u][v]['rank'] for u,v in graph.edges]
            used_rank = True
        except:
            edge_color = [graph[u][v]['rung'] for u,v in graph.edges]
    else:
        edge_color = [graph[u][v]['rung'] for u,v in graph.edges]
    nx.draw(graph, pos, ax=axs[0], with_labels=False, arrows=True, node_size=0, font_size=9, edge_color=edge_color, edge_cmap=edge_cmap)

    node_colors = [v for v in nx.get_node_attributes(graph, 'num_ancestors').values()]
    
    nx.draw_networkx_nodes(graph, pos=pos, node_size=0)
    
    labels = nx.draw_networkx_labels(graph, pos=pos, ax=axs[0],
                                    labels={node:node for node in graph.nodes.keys()},
                                    bbox=dict(edgecolor='black', boxstyle='round,pad=0.5'),
                                    font_size=label_font_size)

    # node color
    color_scale = 0.2
    normalize_node_colors = (np.array(node_colors) - min(node_colors))/(max(node_colors) - min(node_colors))
    for t, c in zip(labels.values(), normalize_node_colors):
        #manipulate indiviual text objects
        t.set_backgroundcolor(node_cmap(c-color_scale))
    
    # plot node colorbar (number of ancestors (ie. larger molecules))
    norm = colors.Normalize(min(node_colors), max(node_colors), clip=True)
    cbar = axs[0].figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=node_cmap), pad=0.05, cax=axs[1])
    cbar.set_label(label=f'Number of Dependents', size=20)
    cbar.ax.tick_params(labelsize=20)

    # plot edge colorbar (rung)
    edge_attr = 'rung'
    if used_rank:
        edge_attr = 'rank'
    edge_cmap_bounds = list(set(nx.get_edge_attributes(graph, edge_attr).values()))
    if len(edge_cmap_bounds) == 1:
        edge_cmap_bounds = [edge_cmap_bounds[-1]-1] + edge_cmap_bounds
        edge_cmap_bounds.append(edge_cmap_bounds[-1]+1)
    edge_cmap_bounds = np.array(edge_cmap_bounds)
    edge_norm = colors.BoundaryNorm(edge_cmap_bounds, edge_cmap.N)
    cbar = axs[1].figure.colorbar(plt.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap), pad=0.05, cax=axs[2])
    if used_rank:
        cbar.set_label(label='Level of Theory', size=20)
    else:
        cbar.set_label(label='CBH Rung', size=20)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    if save_fig_path and not dpi:
        plt.savefig(save_fig_path)
    elif save_fig_path and dpi:
        plt.savefig(save_fig_path, dpi=dpi)
    plt.show()