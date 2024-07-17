import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import math
import random
from scipy.optimize import minimize_scalar
import networkx as nx

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

def get_periods_list(
    timestamps: pd.Series,
    start_day=0,
    duration=3,
    n_periods=13,
):
    """Get a list of periods for a specified amount of days in the dataframe. Also returns the stop days as a list."""
    days_int = np.linspace(start_day, start_day + duration, n_periods)
    days_td = [pd.Timedelta(days=d) for d in days_int]
    day0 = timestamps[0]
    start = day0 + pd.Timedelta(days=start_day)
    periods_list = []
    for days in days_td[1:]:
        stop = day0 + days
        periods_list.append((start, stop))
    return periods_list, days_int[1:]

def get_training_periods(
    timestamps: pd.Series,
    start : pd.DatetimeIndex,
    stop : pd.DatetimeIndex,
    time_step=1.0,
    duration=1.0,
):
    """Get stratified periods in a certain date range. Input 'time_step' and 'duration' are given in days."""
    duration = pd.Timedelta(days=duration)
    time_step = pd.Timedelta(days=time_step)
    periods = []
    sub_start = start
    sub_stop = sub_start + duration
    while sub_stop < stop:
        periods.append((sub_start, sub_stop))
        sub_start += time_step
        sub_stop += time_step
    return periods