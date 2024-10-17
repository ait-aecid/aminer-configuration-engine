import os
import pandas as pd
import numpy as np
import json
import shutil
from tqdm import tqdm

# import from submodule
import sys
sys.path.append('log-preprocessor')
from utils.constants import *

from lib.utils import *

def copy_and_save_file(input_file, output_file, line_numbers):
    """Write specified lines of the input file into the output file."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        for line_number in line_numbers:
            if 0 <= line_number < len(lines):
                outfile.write(lines[line_number])

def get_results_file(filename='/tmp/aminer_out.json', save=False, save_to="tmp/aminer_out.json"):
    """Get output file that was generated from running the AMiner. JSON format is expected ('pretty=false')."""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    if save:
        shutil.copy(filename, save_to)
    return data
        
def get_relevant_info(detectors) -> dict:
    """Returns detector type, id of the triggered instance, line index, timestamp and variable(s) for each alert."""
    results = get_results_file()
    info = []
    for detector in detectors:
        for i in range(len(results)):
            if results[i]['AnalysisComponent']["AnalysisComponentType"].startswith(detector):
                var = results[i]['AnalysisComponent']["AffectedLogAtomPaths"]
                idx = results[i]["LineNumber"]
                ts = pd.to_datetime(results[i]['LogData']["Timestamps"], unit="s")
                crit = results[i]['AnalysisComponent']["CriticalValue"] if "CriticalValue" in results[i]['AnalysisComponent'].keys() else None
                id = results[i]['AnalysisComponent']["AnalysisComponentName"]
                info.append({"detector":detector, "id":id, "var":var, "idx":idx, "ts":ts, "crit":crit})
    return pd.DataFrame(info, columns=["detector", "id", "var", "idx", "ts", "crit"])


class Optimization:

    def aminer_run(self, X, analysis_config, training: bool, label: str, optimization_run=False):
        """Fit AMiner to training data and predict test data."""
        if optimization_run:
            opt_dir = "optimization/"
        else:
            opt_dir = ""
        outputfile = os.path.join(self.output_dir, opt_dir, label + "_data.log")
        config_path = os.path.join(self.output_dir, opt_dir, label + "_config.yaml")
        # update config
        self.config["LearnMode"] = training
        self.config["LogResourceList"] = [os.path.join(self.current_dir, self.data_path)]
        self.config["Analysis"] = analysis_config
        # save config file
        dump_config(config_path, self.config)
        # save data for aminer
        copy_and_save_file(self.data_path, outputfile, list(X.index))
        # run AMiner
        if training:
            command = "sudo aminer -C -o -c " + config_path
        else:
            command = "sudo aminer -o -c " + config_path
        os.system(command)

    def optimization(
        self, 
        X: pd.DataFrame, 
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
        os.system("sudo echo")
        if detectors == "all":
            detectors = list(DETECTOR_ID_DICT.values())
        # if no detectors specified
        elif isinstance(detectors, list) and len(set(detectors).intersection(set(self.detectors))) == 0:
            return analysis_config
        detectors_opt = list(set(detectors).intersection(set(self.detectors)))
        # extract thresh settings
        thresh_names = [setting["parameter_name"] for setting in thresh_optimization.values()]
        n_samples = len(X)
        fold_size = n_samples // (k+1)
        if fancy:
            splits = tqdm(range(1,k+1), desc='Optimizing configuration', unit='iteration', ncols=100)
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
            label = str(i)
            start, end = i * fold_size, (i + 1) * fold_size
            X_train = pd.concat([X.iloc[:start]])
            X_test = X.iloc[start:end]
            split_sizes.append(len(X_train))
            # RUN AMINER
            self.aminer_run(X_train, opt_config, True, "train" + label, optimization_run=True)
            self.aminer_run(X_test, opt_config, False, "test" + label, optimization_run=True)
            # get results and extract relevant infos
            info = get_relevant_info(detectors)
            for id in all_ids:
                df_id = info[info["id"]==id]
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