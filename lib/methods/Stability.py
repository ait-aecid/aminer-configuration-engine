import numpy as np
import pandas as pd

def get_charset_length_evolution(df: pd.DataFrame, by_occurrence=False, relative=False):
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

def get_numeric_variables(df: pd.DataFrame):
        """Returns the variables that can be converted to numeric type."""
        numeric_variables = [var for var in df.columns if all(pd.to_numeric(df[var], errors="coerce").notna())]
        return numeric_variables

def get_numeric_range_evolution(df: pd.DataFrame):
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

def is_stable(changes_list, segment_threshs: list):
    """Determine if a list of segment means is stable."""
    # define averaging operation
    op = lambda y: list(map(lambda x : np.mean(x) if len(x) > 0 else np.nan, y))
    segments = np.array_split(changes_list, len(segment_threshs))
    segment_means = op(segments)
    return all([not q >= thresh for q, thresh in zip(segment_means, segment_threshs)])


def get_unique_sequence_evolution(df: pd.DataFrame, sequence_length=2, diff=True, full_paths_list=False):
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

def get_stable_unique_sequence_evolution(df: pd.DataFrame, segment_threshs, max_length=10, full_paths_list=False):
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

class Stability:
    def get_stable_variables(self, segment_threshs=[1.0, 0.165, 0.027, 0.005, 0.001], how="by_occurrence", group_variables=False):
        """Returns the variables that were classifed as stable."""
        # choose options
        if how == "by_charset":
            changes_dict = get_charset_length_evolution(self.df, relative=True)
        elif how == "by_occurrence":
            changes_dict = get_unique_occurrences(self.df, accumulated=False)
        elif how == "by_valueRange":
            changes_dict = get_numeric_range_evolution(self.df)
        elif how == "by_eventSequence":
            stable_vars_sequence_length_dict = get_stable_unique_sequence_evolution(self.df, segment_threshs)
            stable_vars_sequence_length_dict.update(get_stable_unique_sequence_evolution(self.df, segment_threshs, full_paths_list=True))
            self.event_sequence_lengths = stable_vars_sequence_length_dict
            stable_vars = list(stable_vars_sequence_length_dict.keys())
            return stable_vars
        else:
            raise ValueError(f"Option 'how={how}' not supported.")
        stable_vars = [var for var in changes_dict.keys() if is_stable(changes_dict[var], segment_threshs)]
        if group_variables:
            return [stable_vars]
        else:
            return stable_vars
        
    def get_event_sequence_length(self, parameter_name):
        """Return event sequence lengths."""
        return {parameter_name: self.event_sequence_lengths}