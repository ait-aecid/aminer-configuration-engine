import pandas as pd

def get_unique_values_number(df: pd.DataFrame):
    """Get a dictionary with the count of individual match elements as values and paths as keys."""
    return df.nunique().to_dict()

def get_occurrences_per_value(df: pd.DataFrame):
    """Return the number of occurrences per value for each variable."""
    occurrences_per_value = {}
    for var in df.columns:
        dense_df = df[var].dropna()
        val_count = {val : dense_df.eq(val).sum() for val in set(dense_df)}
        occurrences_per_value[var] = val_count
    return occurrences_per_value

def get_vars_with_random_values(df: pd.DataFrame, thresh=2):
    """Returns variables that contain random values."""
    randoms = []
    occurrences_per_value = get_occurrences_per_value(df)
    for var in occurrences_per_value.keys():
        if occurrences_per_value[var]:
            if min(occurrences_per_value[var].values()) < thresh:
                randoms.append(var)
    return randoms


class Static:
    def get_static_variables(self, thresh=1):
        """Get static variables of a dataframe."""
        unique_values_count = get_unique_values_number(self.df)
        statics = [var for var in self.df.columns if unique_values_count[var] <= thresh]
        return statics

class Random:
    def get_random_variables(self, min_value_occurrence=2):
        """Get variables that occurr randomly."""
        #randoms = [var for var in self.df.columns if self.unique_values_per_variable[var] >= len(self.df[var].dropna()) - thresh]
        randoms = []
        if min_value_occurrence > 0:
            randoms += get_vars_with_random_values(self.df, min_value_occurrence)
        return list(set(randoms))
    
class MinMaxOccurence:
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

class Selection:
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
