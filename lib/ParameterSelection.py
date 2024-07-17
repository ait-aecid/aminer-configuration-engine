import pandas as pd
import numpy as np
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.signal import find_peaks
import statsmodels.api as sm

from lib.transformationUtils import *
from lib.selectionUtils import *

class ParameterSelection:
    """This class handles the selection of parameters."""

    def __init__(self, df, settings={}):       
        self.df_original = df.copy()
        self.df = df.copy()
        self.unique_values_per_variable = self.get_unique_values_number()

        # map meta-parameters to functions
        mapping = {
            "Variables" : {
                "Selected" : self.get_selected_variables,
                "Occurrence" : self.get_variables_by_occurrence,
                "Static" : self.get_static_variables,
                "Random" : self.get_random_variables,
                "Stable" : self.get_stable_variables,
                "Numeric" : self.get_numeric_variables,
                "CharacterPairProbability" : self.get_variables_by_charPairProb,
                "Co-OccurrenceCombos" : self.get_combos_by_co_occurrence,
                "EventFrequency" : self.event_frequency_analysis,
            },
            "SpecificParams" : {
                "CharacterPairProbability" : self.get_charPairProb_thresh,
                "EventFrequencyParams": self.get_event_frequency_params,
            }
        }
        
        #print(self.unique_values_per_variable)

        if settings: # if not empty
            # map meta-parameters to functions
            params = settings.copy()
            self.selected_vars = []
            v = "Variables"
            for action in ["PreFilter", "Select", "PostFilter"]: #TO-DO: implement "PostFilter"
                if action in settings[v].keys():
                    for method in settings[v][action].keys():
                        variables = mapping[v][method](**settings[v][action][method])
                        params[v][action][method] = variables
                        if action == "PreFilter":
                            self.df = self.df.drop(columns=variables)
                        elif action == "Select":
                            self.selected_vars += variables

            # all_variables = list(self.df.columns)
            # for action in settings[v].keys():
            #     method = list(settings[v][action].keys())[0]
            #     method_variables = mapping[v][method](**settings[v][action][method])
            #     params[v][action][method] = method_variables
            #     if action == "Filter":
            #         self.df = self.df.drop(columns=method_variables)
            #     elif action == "Select":
            #         self.selected_vars += method_variables
                            
            s = "SpecificParams"
            if s in settings.keys():
                for method in settings[s].keys():
                    specific_param = mapping[s][method](**settings[s][method])
                    params[s][method] = specific_param
            self.params = params

    def get_event_frequency_params(self):
        """Return EventFrequencyDetector parameters."""
        return self.ef_params

    def event_frequency_analysis(
        self,
        season_min_autocorrelation=0.3,
        season_corr_stepsize=0.033, 
        season_min_reps=3,
        season_variation_coeff=0.1,
        season_smoothing_iterations=10,
        season_planB=None, # e.g. 24h - 86400
        events_per_window=1, # TO-DO: change name
        unique_events_per_window=True,
        operation="median",
    ):
        """Perform analysis for EventFrequencyDetector."""
        seasonality = {}
        window_size = {}
        #confidence_factor = {}
        time_series_df = self.get_time_series_df()
        for var in time_series_df.columns:
            ###### window size ######
            event_timestamps = self.df[[var, "ts"]].dropna(how="any")["ts"]
            if len(set(event_timestamps)) < 2: # at least 2 (unique) events are necessary
                continue
            event_timediffs = pd.to_timedelta(np.diff(event_timestamps)).total_seconds()
            # compute window features
            if unique_events_per_window: # get rid of zeros to treat multiple events at same time as one
                event_timediffs = [e for e in event_timediffs if e != 0]
            if operation == "median":
                timediffs_op = np.median(event_timediffs) 
            elif operation == "mean":
                timediffs_op = np.mean(event_timediffs)
            else:
                raise ValueError("None of the options matched.")
            window_size[var] = int(timediffs_op * events_per_window)

            ###### season #######
            rolling_window = 60 # start with 1 minute time window
            for iter in range(season_smoothing_iterations + 1):
                time_series = time_series_df[var]
                time_series = time_series.rolling(f"{rolling_window}s").mean() # apply rolling average to autocorrelation
                autocorrelations = sm.tsa.acf(time_series, nlags=len(time_series)-1, adjusted=True)
                # remove last 10% because they are usually broken with "adjusted=True"
                autocorrelations = autocorrelations[:-int(len(autocorrelations) * 0.1)]
                corr_broken_idx = np.where(abs(autocorrelations) > 1)[0]
                if len(corr_broken_idx) > 0.4 * len(autocorrelations): # just a test - replace by more sophisticated
                    raise ValueError("Something went wrong when computing correlations.")
                autocorrelations = np.delete(autocorrelations, corr_broken_idx)

                max_corr = 1
                season_found = False
                # scan for peaks
                while max_corr > season_min_autocorrelation and not season_found:
                    peaks, _ = (find_peaks((autocorrelations), height=max_corr, prominence=0.1))
                    peaks = np.insert(peaks, 0, 0)
                    if len(peaks) >= season_min_reps:
                        diffs = pd.to_timedelta(np.diff(time_series.index[peaks]))
                        season_mean = np.mean(diffs.total_seconds())
                        season_std = np.std(diffs.total_seconds())
                        #print("Coeff of variation:", season_std / season_mean)
                        if season_std / season_mean < season_variation_coeff:  #coeff of variation
                            seasonality[var] = int(season_mean)
                            season_found = True
                    max_corr -= season_corr_stepsize
                if season_found:
                    break
                else:
                    # set rolling window and try again
                    rolling_window = int(round(0.05*len(time_series)*60*iter, 0)) # i am a genius
            if var not in seasonality.keys() and season_planB != None:
                seasonality[var] = season_planB

        # only take the one with non-zero window size
        seasonality = {key: val for key, val in seasonality.items() if key in window_size.keys() and window_size[key] > 0}
        window_size = {key: window_size[key] for key, val in seasonality.items()}
        
        self.ef_params = {
            "window_size": window_size,
            "num_windows": {var: 300 for var in seasonality.keys()},
            "confidence_factor": {var: 0.2 for var in seasonality.keys()},
            "season": seasonality,
            "empty_window_warnings" : {var: False for var in seasonality.keys()}
        }
        return list(seasonality.keys())

    def get_time_series_df(self, sampling="min", drop_duplicates=True):
        """Get the data as a df of time series of occurrences."""
        time_series_df = self.df.set_index("ts").apply(lambda x: x.resample(sampling).count())
        if drop_duplicates:
            time_series_df = time_series_df.T.drop_duplicates().T
        return time_series_df
    
    def get_variables_by_occurrence(self, thresh: int, how: str, rel=True):
        """Get variables by how often they occur."""
        n = len(self.df)
        if not rel:
            n = 1
        occurrence_dict = self.df.notna().sum().to_dict()
        if how == "less_than":
            return [key for key, val in occurrence_dict.items() if val <= thresh*n]
        elif how == "more_than":
            return [key for key, val in occurrence_dict.items() if val >= thresh*n]
        else:
            raise ValueError(f"Option {how} not available. Choose either 'less' or 'more'.")

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
    
    def get_charPairProbs(self, variables=[]) -> dict:
        """Returns the probabilities (critical values) for each value of each variable.
        Calculated from the probabilities of their character pairs."""
        if variables == "all":
            variables = self.df.columns
        df = self.df[variables].applymap(lambda x: str(x) if not pd.isna(x) else x)
        critical_values = {}
        for var in df.columns:
            value_set = {}
            freq = {}
            total_freq = {}
            critical_values[var] = []
            for value in df[var]:
                if pd.isna(value): # skip nan values
                    continue
                probs = []
                for i in range(-1, len(value)):
                    first_char = -1
                    if i != -1:
                        first_char = value[i]
                    second_char = -1
                    if i != len(value) - 1:
                        second_char = value[i + 1]
                    if first_char in freq:
                        total_freq[first_char] += 1
                        if second_char in freq[first_char]:
                            freq[first_char][second_char] += 1
                        else:
                            freq[first_char][second_char] = 1
                    else:
                        total_freq[first_char] = 1
                        freq[first_char] = {}
                        freq[first_char][second_char] = 1
                    prob = 0
                    if first_char in freq and second_char in freq[first_char]:
                        prob = freq[first_char][second_char] / total_freq[first_char]
                    probs.append(prob)
                critical_val = sum(probs) / len(probs)
                critical_values[var].append(critical_val)
                value_set[value] = critical_val
        critical_values = {key: val for key, val in critical_values.items() if len(val) > 0}
        return critical_values

    def get_variables_by_charPairProb(self, mean_crit_thresh=0.6):
        """Return variables that have a mean character pair probability above the threshold."""
        self.critical_values = self.get_charPairProbs("all")
        means = {key: np.mean(val) for key, val in self.critical_values.items()}
        selection = [key for key, val in means.items() if val >= mean_crit_thresh]
        return selection
    

    def get_charPairProb_thresh(self, parameter_name="", min=0, max=1, offset=0.05):
        """Returns minimum of the probabilities of the character pairs for each variable."""
        if not self.critical_values: # if not defined yet
            self.critical_values = self.get_charPairProbs("all")
        min_max_round = lambda x: float(round(np.min([np.max([np.max([np.min(x) + offset, 0]), min]), max]), 3))
        critical_values_min = {key : min_max_round(val) for key, val in self.critical_values.items()}
        return {parameter_name: critical_values_min}


    def get_selected_variables(self, paths="all"):
        if type(paths) == list:
            return paths
        elif paths == "all":
            return list(self.df.columns)
        # for the case that some variables were already pre-filtered
        elif paths == "original": 
            return list(self.df_original)
        else:
            raise ValueError(f"Option '{paths}' not available.")


    def get_unique_values_number(self):
        """Get a dictionary with the count of individual match elements as values and paths as keys."""
        return self.df.nunique().to_dict()


    def get_static_variables(self, thresh=1):
        """Get static variables of a dataframe."""
        statics = [var for var in self.df.columns if self.unique_values_per_variable[var] <= thresh]
        return statics


    def get_random_variables(self, thresh=3, min_value_occurrence=2):
        """Get variables that occurr randomly."""
        #randoms = [var for var in self.df.columns if self.unique_values_per_variable[var] >= len(self.df[var].dropna()) - thresh]
        randoms = []
        if min_value_occurrence > 0:
            randoms += self.get_vars_with_random_values(min_value_occurrence)
        return list(set(randoms))


    def get_unique_occurrence_segment_means(df : pd.DataFrame, n_segments, how="by_occurrence"):
        """Compute some measure of stability for each variable. Returns the means of each segment."""
        op = lambda y: list(map(lambda x : np.mean(x) if len(x) > 0 else np.nan, y))
        segments_per_var = {}
        if how == "by_charset":
            counts_dict = ParameterSelection.get_charset_length_evolution(df, relative=True)
        elif how == "by_occurrence":
            counts_dict = ParameterSelection.get_unique_occurrences(df, accumulated=False)
        elif how == "by_valueRange":
            counts_dict = ParameterSelection.numeric_range_evolution(df)
        else:
            raise ValueError(f"Option '{how}' not supported. Choose 'by_occurrence', 'by_charset' or 'by_valueRange'.")
        for var, counts in zip(counts_dict.keys(), counts_dict.values()):
            segments = np.array_split(counts, n_segments)
            segment_means = op(segments)
            segments_per_var[var] = segment_means
        return segments_per_var

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
        
    
    def get_stable_variables(self, segment_threshs, how="by_occurrence", group_variables=False):
        """Returns the variables that were classifed as stable."""
        is_unstable = lambda q_list : not all([not q >= thresh for q, thresh in zip(q_list, segment_threshs)])
        segments = ParameterSelection.get_unique_occurrence_segment_means(self.df, len(segment_threshs), how)
        stable_vars = [var for var in segments.keys() if not is_unstable(segments[var])]
        if group_variables:
            return [stable_vars] # some detectors are to inefficient with too many instances
        else:
            return stable_vars


    def get_occurrences_per_value(df: pd.DataFrame):
        """Return the number of occurrences per value for each variable."""
        occurrences_per_value = {}
        for var in df.columns:
            dense_df = df[var].dropna()
            val_count = {val : dense_df.eq(val).sum() for val in set(dense_df)}
            occurrences_per_value[var] = val_count
        return occurrences_per_value
    

    def get_vars_with_random_values(self, thresh=2):
        """Returns variables that contain random values."""
        randoms = []
        occurrences_per_value = ParameterSelection.get_occurrences_per_value(self.df)
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
    
    def get_numeric_variables(df):
        """Returns the variables that can be converted to numeric type."""
        numeric_variables = [var for var in df.columns if all(pd.to_numeric(df[var], errors="coerce").notna())]
        return numeric_variables
    
    def numeric_range_evolution(df):
        """Returns a dict of numeric variables as keys and lists as values whose are 1 if the min-max range was extended, 0 otherwise."""
        numeric_vars = ParameterSelection.get_numeric_variables(df)
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

    
    def get_duplicates(arr: list):
        """Returns the duplicates of a list."""
        unique_elements, counts = np.unique(arr, return_counts=True)
        duplicates = unique_elements[counts > 1]
        return list(duplicates)
    
