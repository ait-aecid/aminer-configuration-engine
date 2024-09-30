import pandas as pd
import numpy as np
import itertools
import os
import yaml
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import math
import networkx as nx

def copy_and_save_file(input_file, output_file, line_numbers):
    """Write specified lines of the input file into the output file."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        for line_number in line_numbers:
            if 0 <= line_number < len(lines):
                outfile.write(lines[line_number])

def concatenate_files(input_files, output_file):
    """Concatenate multiple files into one file."""
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                outfile.write(infile.read())

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Nomen est omen."""
    dict_of_lists = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists

def get_timestamps(data : pd.DataFrame, predefined_timestamp_paths):
    """Returns timestamps of the data."""
    timestamps_paths = [ts for ts in predefined_timestamp_paths if ts in data.columns]
    ts_series = data[timestamps_paths].ffill(axis=1).bfill(axis=1).iloc[:, 0]
    ts_series.name = "timestamps"
    try:
        return pd.to_datetime(ts_series, unit="s") # audit 
    except:
        return pd.to_datetime(ts_series, format="%d/%b/%Y:%H:%M:%S %z").dt.tz_localize(None) # apache

def encode_df(df) -> pd.DataFrame:
    """Encode each column of a df individually."""
    return df.apply(lambda x: LabelEncoder().fit_transform(x))

def filter_by_value_occurrence(series: pd.Series, thresh=2):
    """Filter values that occur less than the threshold."""
    value_counts = series.value_counts()
    filtered_values = value_counts[value_counts >= thresh].index
    filtered_series = series[series.isin(filtered_values)]
    return filtered_series

def get_combos(arr, max_combo_length=2):
    """Get all possible combinations of a list."""
    combos = []
    for i in range(2, min(len(arr), max_combo_length, 5) + 1):
        combo = list(combinations(arr, i))
        combos.extend(combo)
    return combos

def get_number_of_combos(n, k_min=2, k_max=3):
    """Calculate the number of possible combinations."""
    n = int(n)
    number_of_combos = 0
    for i in range(k_min, k_max + 1):
        number_of_combos += math.comb(n, i)
    return number_of_combos

def merge_connected_elements(tuples_list: list):
    """Merge connected combinations if they are connected."""
    G = nx.Graph()
    G.add_edges_from(tuples_list)
    connected_components = nx.connected_components(G)
    merged_chains = []
    for component in connected_components:
        merged_chains.append(tuple(sorted(component)))
    return merged_chains

def merge_cycles(tuples_list: list):
    """Merge connected combinations if they form a cycle."""
    G = nx.Graph()
    G.add_edges_from(tuples_list)
    merged = []
    cycles = nx.cycle_basis(G)
    for cycle in cycles:
        if len(cycle) >= 3:
            merged.append(tuple(sorted(cycle)))
    return merged

def get_weighted_dict(unweighted_values : dict, weights : dict, complementary_percentage=False):
    """Weigh the values of a dictionary by weights defined in another dictionary. Keep in mind that their keys have to match."""
    if complementary_percentage: 
        weighted_list = [unweighted_values[key]*(1-weights[key]) for key in unweighted_values.keys()]
    else: 
        weighted_list = [unweighted_values[key]*(weights[key]) for key in unweighted_values.keys()]
    weighted_dict = {
        key: value for key, value in zip(unweighted_values.keys(), weighted_list)
    }
    return weighted_dict

def get_unique_occurrences(df : pd.DataFrame, accumulated=False, uniques_given=None, return_uniques=False):
    """Return a dictionary with variables as keys and the numbers of unique occurrences per occurrence (lists) as values."""
    counts_dict = {}
    uniques_dict = {}
    for var in df.columns:
        if return_uniques and uniques_given != None:
            uniques = uniques_given[var]
        else:
            uniques = set()
        counts = []
        dense_df = df[var].dropna()
        for val in dense_df:
            if val not in uniques:
                uniques.add(val)
                counts.append(1)
            else:
                counts.append(0)
        uniques_dict[var] = uniques
        if accumulated:
            counts = np.cumsum(counts)
        counts_dict[var] = counts
    if return_uniques:
        return counts_dict, uniques_dict
    else:
        return counts_dict

def get_occurrences_per_value(df: pd.DataFrame):
    """Return the number of occurrences per value for each variable."""
    occurrences_per_value = {}
    for var in df.columns:
        dense_df = df[var].dropna()
        val_count = {val : dense_df.eq(val).sum() for val in set(dense_df)}
        occurrences_per_value[var] = val_count
    return occurrences_per_value

def get_vars_with_random_values(df, thresh=2):
    """Returns variables that contain random values."""
    randoms = []
    occurrences_per_value = get_occurrences_per_value(df)
    for var in occurrences_per_value.keys():
        if occurrences_per_value[var]:
            if min(occurrences_per_value[var].values()) < thresh:
                randoms.append(var)
    return randoms

def get_charset_length_evolution(df, by_occurrence=False, relative=False):
    """Get the evolution of the length of the charsets of each variable."""
    charset_lengths_dict = {}
    for var in df.columns:
        charset = set()
        charset_lengths = [0]
        for val in df[var]:
            if pd.isna(val) and by_occurrence:
                continue
            charset = charset.union(set(str(val)))
            charset_lengths.append(len(charset))
        if relative:
            diffs = np.diff(charset_lengths, prepend=0)
            charset_lengths_dict[var] = list(np.where(diffs != 0, 1, diffs))
        else:
            charset_lengths_dict[var] = charset_lengths
    return charset_lengths_dict

def get_numeric_range_evolution(df):
    """Returns a dict of numeric variables as keys and lists as values which are 1 if the min-max range was extended, 0 otherwise."""
    numeric_vars = get_numeric_variables(df)
    df_num = df[numeric_vars].astype(float)
    range_ext_dict = {}
    for var in df_num.columns:
        range_extension = []
        max_value, min_value = 0, 0
        for val in df_num[var].dropna():
            if val > max_value:
                max_value = val
                range_extension.append(1)
            elif val < min_value:
                min_value = val
                range_extension.append(1)
            else:
                range_extension.append(0)
        range_ext_dict[var] = range_extension
    return range_ext_dict

def get_numeric_variables(df):
        """Returns the variables that can be converted to numeric type."""
        numeric_variables = [var for var in df.columns if all(pd.to_numeric(df[var], errors="coerce").notna())]
        return numeric_variables

def get_unique_values_number(df):
    """Get a dictionary with the count of individual match elements as values and paths as keys."""
    return df.nunique().to_dict()

def is_stable(changes_list, segment_threshs):
    """Determine if a list of segment means is stable."""
    # define averaging operation
    op = lambda y: list(map(lambda x : np.mean(x) if len(x) > 0 else np.nan, y))
    segments = np.array_split(changes_list, len(segment_threshs))
    segment_means = op(segments)
    return all([not q >= thresh for q, thresh in zip(segment_means, segment_threshs)])

def get_unique_sequence_evolution(df, sequence_length=2, diff=True, full_paths_list=False):
    """Get the evolution of unique sequences in a DataFrame over time."""
    unique_sequences_evolution = {}
    # take full paths list if true
    iteration_list = df.columns if not full_paths_list else [None]
    # variable 'None' corresponds to full paths list
    for var in iteration_list:
        if not full_paths_list:
            events = df[var].values
        else:
            # get full paths list
            events = [tuple(df.loc[i].dropna().index) for i in range(len(df))]
        n_events = len(events)
        if n_events < sequence_length:
            continue
        sequences = set()
        unique_sequence_count = np.zeros(n_events - sequence_length, dtype=int)
        for i in range(n_events - sequence_length):
            sequence = tuple(events[i:i + sequence_length]) 
            unique_sequence_count[i] = len(sequences)
            sequences.add(sequence)
        unique_sequences_evolution[var] = np.diff(unique_sequence_count, prepend=0) if diff else unique_sequence_count
    return unique_sequences_evolution

def get_stable_unique_sequence_evolution(df, segment_threshs, max_length=10, full_paths_list=False):
    """Get the variables and the largest lengths for which they are stable by the evolution of unique sequences."""
    df_ = df.copy()
    stable_vars_lengths = {}
    for length in range(2, max_length):
        # takes the full paths list instead of variables if 'full_paths_list' == True
        unique_sequence_evo = get_unique_sequence_evolution(df_, length, full_paths_list=full_paths_list)
        stable_vars = [var for var in unique_sequence_evo.keys() if is_stable(unique_sequence_evo[var], segment_threshs)]
        if len(stable_vars) == 0:
            break
        stable_vars_dict = {key: int(val) for key, val in zip(stable_vars, np.full(len(stable_vars), length))}
        stable_vars_lengths.update(stable_vars_dict)
        df_ = df_[stable_vars] if not full_paths_list else df_
    return stable_vars_lengths