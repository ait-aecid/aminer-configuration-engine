import pandas as pd
import numpy as np
import itertools
import os
import yaml
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import math
import networkx as nx

def set_without_NaN(data):
    """Create a set of a listlike object where NaN values are excluded."""
    set_ = set(data)
    if np.NaN in set_:
        set_.remove(np.NaN)
    return set_

def sort_dict_by_values(dictionary: dict, reverse=False):
    """Sort a dictionary by its values."""
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=reverse))

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

def represent_float(dumper, data):
    if data != data:  # check for NaN
        value = '.nan'
    elif data == float('inf'):
        value = '.inf'
    elif data == float('-inf'):
        value = '-.inf'
    else:
        value = float(data)
    return dumper.represent_scalar('tag:yaml.org,2002:float', value)

def convert_to_intervals(integer_list):
    """Convert elements of a list into intervals."""
    intervals = []
    for i in range(0, len(integer_list)-1):
        start = integer_list[i]
        end = integer_list[i + 1]
        intervals.append((start, end))
    return intervals

def find_overlapping_intervals(interval_list, target_interval):
    """Returns a list of booleans indicating overlapping."""
    overlapping_intervals = []
    for interval in interval_list:
        overlapping_intervals.append(interval[1] >= target_interval[0] and interval[0] <= target_interval[1])
    return overlapping_intervals

def group_consecutive(input_list: list) -> list:
    groups = []
    for k, g in itertools.groupby(enumerate(input_list), lambda x: x[0] - x[1]):
        groups.append(list(map(lambda x: x[1], g)))
    return [(group[0], group[-1]) for group in groups]

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Nomen est omen."""
    dict_of_lists = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists

def calculate_metrics(df):
    """Calculate the evaluation metrics and add to df."""
    precision = df['TP'] / (df['TP'] + df['FP'])
    recall = df['TP'] / (df['TP'] + df['FN'])
    accuracy = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])
    balanced_accuracy = (recall + df["TN"] / (df["TN"] + df["FP"])) / 2
    true_positive_rate = df['TP'] / (df['TP'] + df['FN'])
    false_positive_rate = df['FP'] / (df['FP'] + df['TN'])
    #auc = metrics.auc(false_positive_rate, true_positive_rate)
    f1_score = 2 * (precision * recall) / (precision + recall)

    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score,
        'Accuracy': accuracy,
        'Accuracy balanced': balanced_accuracy,
        'TP-Rate': true_positive_rate,
        'FP-Rate': false_positive_rate,
        #'AUC': auc,
    })
    combined_df = pd.concat([df, metrics_df], axis=1)
    return combined_df

def metrics_dicts_to_df(eval_dicts: list, nan_to_zero=True, to_csv=False, path="results", filename="", hp_list=None):
    """Turn a list of dicts of dicts into a df."""
    restructured_results = list_of_dicts_to_dict_of_lists(eval_dicts)
    df_dict = {}
    for key in restructured_results.keys():
        df_dict[key] = calculate_metrics(pd.DataFrame(restructured_results[key]))
        if nan_to_zero:
            df_dict[key] = df_dict[key].fillna(0)
        if hp_list is not None:
            df_dict[key]["hp"] = hp_list
        if to_csv:
            if not os.path.exists(path):
                os.makedirs(path)
            print("Saving to:", path)
            df_dict[key].to_csv(f"{path}/{key}_{filename}.csv", index=False)
    return df_dict

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