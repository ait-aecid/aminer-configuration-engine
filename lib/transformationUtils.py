import pandas as pd
import numpy as np
import itertools
import os
#import ruamel.yaml
import yaml
from sklearn import metrics

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