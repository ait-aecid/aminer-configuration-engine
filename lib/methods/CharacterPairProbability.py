import numpy as np
import pandas as pd

class CharacterPairProbability:

    def get_variables_by_charPairProb(self, mean_crit_thresh=0.6):
        """Return variables that have a mean character pair probability above the threshold."""
        self.critical_values = self.get_charPairProbs("all")
        means = {key: np.mean(val) for key, val in self.critical_values.items()}
        selection = [key for key, val in means.items() if val >= mean_crit_thresh]
        return selection
        
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
    
    def get_charPairProb_thresh(self, parameter_name="", min=0, max=1, offset=0.05):
        """Returns minimum of the probabilities of the character pairs for each variable."""
        min_max_round = lambda x: float(round(np.min([np.max([np.max([np.min(x) + offset, 0]), min]), max]), 3))
        critical_values_min = {key : min_max_round(val) for key, val in self.critical_values.items()}
        return {parameter_name: critical_values_min}