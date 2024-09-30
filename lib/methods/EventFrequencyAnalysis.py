import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks

class EventFrequencyAnalysis:
    
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
    
    def get_event_frequency_params(self):
        """Return EventFrequencyDetector parameters."""
        return self.ef_params