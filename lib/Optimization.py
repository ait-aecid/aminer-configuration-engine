import os
import pandas as pd
import numpy as np
import getpass
from tqdm import tqdm

# import from submodule
import sys
sys.path.append('log-preprocessor')
from utils.constants import *
from tools.AMinerModel import AMinerModel

from lib.utils import *

def in_jupyter_notebook():
    """Check if the function was called from within a jupyter notebook."""
    try:
        # Check if the get_ipython function exists (unique to IPython environments)
        from IPython import get_ipython
        ipy_instance = get_ipython()
        if ipy_instance and 'IPKernelApp' in ipy_instance.config:
            return True
        else:
            return False
    except ImportError:
        # IPython is not installed, likely not running in a Jupyter notebook
        return False

class Optimization:
    """This class contains the functionality for optimizing configuration files."""

    def optimize_config(
        self, 
        df: pd.DataFrame, 
        analysis_config: list, 
        detectors: list,
        k=5, 
        #timesplit=False, 
        max_FP=10, 
        max_FP_per_minute=0.1, 
        weighted_split=True,
        thresh_optimization={
            "EntropyDetector": {
                "parameter_name": "prob_thresh",
                "min": 0.01,
                "max": 1,
                "offset": -0.05,
            }
        },            
        fancy=True
    ):
        """Optimize the 'Analysis' part of a configuration."""
        pwd=None
        if in_jupyter_notebook():
            print("(running in Jupyter notebook)")
            if pwd is None:
                pwd = getpass.getpass("Execution in jupyter notebook requires sudo password:")
            os.system(f"echo {pwd} | sudo -S echo -n")
        else:
            os.system("sudo echo -n")
        if detectors == "all":
            detectors = list(DETECTOR_ID_DICT.values())
        # if no detectors specified
        elif isinstance(detectors, list) and len(set(detectors).intersection(set(self.detectors))) == 0:
            print("No detectors specified for optimization. Returning original configuration.")
            return analysis_config
        detectors_opt = list(set(detectors).intersection(set(self.detectors)))
        # extract thresh settings
        thresh_names = [setting["parameter_name"] for setting in thresh_optimization.values()]
        n_samples = len(df)
        fold_size = n_samples // (k+1)
        if fancy:
            splits = tqdm(range(1,k+1), desc='Optimizing configuration', unit='data split', ncols=100)
        else:
            splits = range(1,k+1)
        # remove unspecified detectors from config
        opt_config = [conf for conf in analysis_config if conf["type"] in detectors_opt]
        all_ids = [conf["id"] for conf in opt_config]
        # timesplit-like k-fold cross-validation (split by samples)
        split_sizes = []
        fp_dict = {key: [] for key in all_ids}
        fp_per_minute_dict = {key: [] for key in all_ids}
        crit_min_dict = {key: [] for key in all_ids}
        for i in splits:
            start, end = i * fold_size, (i + 1) * fold_size
            df_train = pd.concat([df.iloc[:start]])
            df_test = df.iloc[start:end]
            split_sizes.append(len(df_train))
            opt_config = self.config.copy()
            opt_config["Analysis"] = analysis_config
            model = AMinerModel(
                config=opt_config,
                input_path=self.tmp_save_path,
                tmp_dir=os.path.join(self.output_dir, "optimization"),
                files_suffix=str(i),
                pwd=pwd
            )
            model.fit_predict(df_train, df_test, print_progress=False)
            model_results = model.get_latest_results_df(detectors)
            for id in all_ids:
                df_id = model_results[model_results["id"]==id]
                fp = len(set(df_id["idx"]))
                timedelta = (max(df_id["ts"])-min(df_id["ts"])).total_seconds().values if not df_id.empty else 0
                seconds = float(timedelta)
                minutes = seconds/60 if seconds != 0 else 1 # minimum is 1 minute
                fp_per_minute = fp/(minutes)
                fp_dict[id].append(fp)
                fp_per_minute_dict[id].append(fp_per_minute)
                min_crit = min(df_id["crit"]) if len(df_id["crit"].dropna()) > 0 else np.nan
                crit_min_dict[id].append(min_crit)
        split_sizes = np.array(split_sizes)
        normalized_weights = split_sizes / sum(split_sizes)
        weighted_mean = lambda x: np.mean([y * i for y, i in zip(x, normalized_weights)])
        optimized_config = analysis_config.copy()
        for id in fp_dict.keys():
            if not weighted_split:
                fp_mean = np.mean(fp_dict[id])
                fp_per_minute_mean = np.mean(fp_per_minute_dict[id])
            else:
                fp_mean = weighted_mean(fp_dict[id])
                fp_per_minute_mean = weighted_mean(fp_per_minute_dict[id])
            fp_condition = fp_mean > max_FP or fp_per_minute_mean > max_FP_per_minute
            if fp_condition:
                for instance in opt_config:
                    if instance["id"] == id:
                        if bool(set(thresh_names) & set(instance.keys())):
                            current_settings = thresh_optimization[instance["type"]]
                            # take minimum of all critical values of all alarms
                            new_thresh = float(round(np.nanmin(crit_min_dict[id]) + current_settings["offset"], 3))
                            # delete instance if thresh outside of allowed range
                            if new_thresh < current_settings["min"] or new_thresh > current_settings["max"]:
                                try:
                                    print(f"Deleting", instance["type"], "-", instance["paths"])
                                except:
                                    print(f"Deleting", instance["type"], "-", instance["constraint_list"])
                                optimized_config.remove(instance)
                            else:
                                # update config with new thresh
                                thresh_name = set(thresh_names).intersection(set(instance.keys())).pop()
                                print(f"Adjusting", instance["type"], "-", instance["paths"], f"... '{thresh_name}': {instance[thresh_name]} -> {new_thresh}")
                                idx = optimized_config.index(instance) # get idx before changing instance
                                instance[thresh_name] = new_thresh
                                optimized_config[idx] = instance
                        else:
                            try:
                                print(f"Deleting", instance["type"], "-", instance["paths"])
                            except:
                                print(f"Deleting", instance["type"], "-", instance["constraint_list"])
                            optimized_config.remove(instance)
        return optimized_config