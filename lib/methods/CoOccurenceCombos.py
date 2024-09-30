import pandas as pd
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import math
import networkx as nx

def get_combos(arr, max_combo_length=2):
    """Get all possible combinations of a list."""
    combos = []
    for i in range(2, min(len(arr), max_combo_length, 5) + 1):
        combo = list(combinations(arr, i))
        combos.extend(combo)
    return combos

def merge_connected_elements(tuples_list: list):
    """Merge connected combinations if they are connected."""
    G = nx.Graph()
    G.add_edges_from(tuples_list)
    connected_components = nx.connected_components(G)
    merged_chains = []
    for component in connected_components:
        merged_chains.append(tuple(sorted(component)))
    return merged_chains


class CoOccurrenceCombos:
    def get_combos_by_co_occurrence(self, min_co_occurrence=0.1, max_combos=40000):
        """Returns promising variable combos based on their co-occurrence."""
        _df = self.df.copy()
        variables = list(_df.columns)
        # encode variables for higher efficiency
        variable_encoder = LabelEncoder()
        _df.columns = variable_encoder.fit_transform(variables)
        # check total number of combos
        length = len(_df.columns)
        number_of_2combos = math.comb(length, 2)
        if number_of_2combos > max_combos:
            print("Number of variables:", length)
            print("Number of 2-combos:", number_of_2combos)
            # some action e.g. add random sampling if number of combos is too high
            raise RuntimeError("Max number of combos exceeded!")
        possible_combos = get_combos(_df.columns, 2)
        preliminary_combos = []
        for combo in possible_combos:
            # assess co-occurrence
            df_co_occurring = _df[list(combo)].dropna(axis=0, how="any").dropna(axis=1, how="all")
            co_occurrence = df_co_occurring.shape[0]
            # compute relative threshold
            var_length = [len(_df[c].dropna()) for c in combo]
            max_count = max(var_length)
            if co_occurrence < max_count * min_co_occurrence:
                continue
            preliminary_combos.append(combo)
        #print(len(preliminary_combos))
        # check if combos can be merged
        final_combos = set()
        for combo in preliminary_combos:
            related_combos = [c for c in preliminary_combos if (combo[0] in c) or (combo[1] in c)]
            merged_combos = merge_connected_elements(related_combos) # merge chained combos using graph theory
            for merged_combo in merged_combos:
                final_combos.add(merged_combo)
        final_combos_decoded = [variable_encoder.inverse_transform(c).tolist() for c in final_combos]
        return final_combos_decoded
    