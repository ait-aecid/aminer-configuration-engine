
import argparse
import os
import pandas as pd
import sys
import time
import numpy as np
import importlib
from tqdm import tqdm

from lib.utils import *
from lib.configUtils import *
from lib.ParameterSelection import *
from lib.Evaluation import *

sys.path.append("/usr/lib/logdata-anomaly-miner")
sys.path.append("/etc/aminer/conf-available/ait-lds")
from aminer.parsing.MatchElement import MatchElement
from aminer.parsing.MatchContext import MatchContext
from aminer.input.LogAtom import LogAtom
from aminer.parsing.ParserMatch import ParserMatch

class AminerConfigurationEngine(ParameterSelection):
    """This class contains all the functionality that is required for the initialization of this project."""

    def __init__(self, params):
        """Initialize project. Returns parsed command line arguments."""

        self.detector_id_dict = {
            "1" : "NewMatchPathValueDetector",
            "2" : "NewMatchPathValueComboDetector",
            "3" : "CharsetDetector",
            "4" : "EntropyDetector",
            "5" : "ValueRangeDetector",
            "6" : "EventFrequencyDetector"
        }

        self.file_type_info = {
            'AuditdParsingModel': {
                'type': "audit",
                'timestamp_format': None,
                'split_char': ["msg=audit(", ":"],
                "unit" : "s",
            },
            'ApacheAccessParsingModel': {
                'type': "apache2/.*access",
                'timestamp_format': '%d/%b/%Y:%H:%M:%S %z',
                'split_char': ["[", "]"],
                "unit": None
            }
        }

        self.label = "TBA"

        # create tmp directory
        os.makedirs("tmp", exist_ok=True)
        os.makedirs(os.path.join("tmp", "data_parsed"), exist_ok=True)

        # get command line arguments if no parameters are passed
        if params == None:
            self.set_and_get_args(False)
        else:
            self.__dict__.update(params)

        self.detectors = [self.detector_id_dict[id] for id in self.detector_ids]

        base_config = get_base_config(self.parser)
        self.timestamp_variables = base_config["Input"]["timestamp_paths"]

        self.data_path = os.path.join("tmp", "current_data.log")
        start = time.time()
        self.df = self.get_data(save_to=self.data_path)
        end = time.time()
        print(f"Finished data extraction (runtime: {end-start}).")

        self.init_output_dir()

        self.current_dir = "file://" + os.getcwd()

        # define standard inputs and fill config dictionaries
        base_config = get_base_config(self.parser)
        self.config = base_config.copy()
        self.timestamp_variables = base_config["Input"]["timestamp_paths"]
        with open("meta-configuration.yaml", 'r') as yaml_file:
            self.settings = yaml.safe_load(yaml_file)


    def logfile_to_df(self, path, interval=None):
        """Get a list of match dictionaries from log data."""
        module = importlib.import_module(self.parser)
        parsing_model = module.get_model()
        # similar to process in /usr/lib/logdata-anomaly-miner/aminer/input/ByteStreamLineAtomizer.py
        match_dict_list = []
        timestamps = []
        # for faster repeated data ingestion
        h5_label = self.data_dir.replace("/","-")
        h5_filename = f"{h5_label}_{self.parser}.h5" # add number of instances
        parsed_data_dir = "tmp/data_parsed/"
        parsed_data_path = os.path.join(parsed_data_dir, h5_filename)
        root, dirs, files = list(os.walk(parsed_data_dir))[0]
        if h5_filename not in files:
            if interval is None:
                print("Parsing data ...")
                with open(path, "rb") as file:
                    for line_data in file:
                        log_atom = LogAtom(line_data, None, None, None)
                        match_context = MatchContext(line_data)
                        match_element = parsing_model.get_match_element("", match_context)
                        if match_element is None:
                            print("Encountered 'None' while parsing.")
                            match_dict_list.append({})
                            timestamps.append(None)
                            continue
                        log_atom.parser_match = ParserMatch(match_element)
                        match_dict = log_atom.parser_match.get_match_dictionary()
                        match_dict_list.append(match_dict)
                        # get ts
                        text_data = line_data.decode("utf-8")
                        timestamps.append(self.get_timestamp_from_string(text_data))
            else:
                with open(path, "rb") as file:
                    for _ in range(interval[0]): # skip lines
                        file.readline()
                    for i in range(interval[0], interval[1]):
                        line_data = file.readline().strip()
                        if not line_data:
                            break
                        log_atom = LogAtom(line_data, None, None, None)
                        match_context = MatchContext(line_data)
                        match_element = parsing_model.get_match_element("", match_context)
                        if match_element is None:
                            print("Encountered 'None' while parsing.")
                            match_dict_list.append({})
                            timestamps.append(None)
                            continue
                        log_atom.parser_match = ParserMatch(match_element)
                        match_dict = log_atom.parser_match.get_match_dictionary()
                        match_dict_list.append(match_dict)
                        # get ts
                        text_data = line_data.decode("utf-8")
                        timestamps.append(self.get_timestamp_from_string(text_data))
            # unwrap contents of match_dict_list from custom class objects (MatchElement) to type string.
            match_dict_list_transformed = [
                dict(map(lambda item: (item[0], item[1].get_match_string().decode()), match_dict.items(),)) for match_dict in match_dict_list
            ]
            df = pd.DataFrame(match_dict_list_transformed)
            df["ts"] = pd.to_datetime(timestamps).tz_localize(None)
            print("Saving parsed data to .h5 file.")
            df.to_hdf(parsed_data_path, key='df', mode='w') 
        else:
            df = pd.read_hdf(parsed_data_path, 'df')
            print("Got parsed data from .h5 file.")
        return df
    
    def get_timestamp_from_string(self, string):
        """Get timestamp from string."""
        split_char = self.file_type_info[self.parser]["split_char"]
        ts_string = string.split(split_char[0])[1].split(split_char[1])[0]
        ts = pd.to_datetime(ts_string, format=self.file_type_info[self.parser]["timestamp_format"], unit=self.file_type_info[self.parser]["unit"])
        return ts

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
        # get sudo password - delete later!!
        with open("/home/viktor/projects/aminer-configuration-engine/key/pwd.txt") as file:
            pwd = file.read()
        # run AMiner
        if training:
            #command = f"echo {pwd} | sudo -S aminer -C -o -c " + config_path
            command = f"sudo aminer -C -o -c " + config_path
        else:
            #command = f"echo {pwd} | sudo -S aminer -o -c " + config_path
            command = f"sudo aminer -C -o -c " + config_path
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
                "name": "prob_thresh",
                "min": 0.01,
                "max": 1,
                "offset": -0.05,
            }
        },            
        fancy=True
    ):
        """Optimize the 'Analysis' part of a configuration."""
        os.system("sudo echo")
        # if no detectors specified
        if len(set(detectors).intersection(set(self.detectors))) == 0:
            return analysis_config
        detectors_opt = list(set(detectors).intersection(set(self.detectors)))
        # extract thresh settings
        thresh_names = [setting["name"] for setting in thresh_optimization.values()]
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
        os.system("sudo echo")
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


    def configure_detectors(self, predefined_config=None) -> list:
        """Configure detectors and return their configurations as dictionaries in a list."""
        df = self.df.copy()
        detector_config = []
        if predefined_config == None:
            for current_detector in self.detectors:
                if current_detector != "EventFrequencyDetector":
                    current_df = df.drop(columns="ts")
                else:
                    current_df = df
                current_analysis = ParameterSelection(current_df, self.settings["ParameterSelection"][current_detector])
                detector_config += assemble_detector(current_detector, current_analysis.params)
        else:
            detector_config = adapt_predefined_analysis_config(predefined_config["Analysis"], self.detectors, df, print_deleted=True)
        
        # give a id if no id is given
        for i, instance in enumerate(detector_config):
            if "persistence_id" not in instance.keys():
                instance["persistence_id"] = f"noName_id_{i}"
                detector_config[i] = instance
            if "id" not in instance.keys():
                instance["id"] = f"noName_id_{i}"
                detector_config[i] = instance

        if len(detector_config) > 0 and self.optimize:
            for split_type in self.settings["Optimization"].keys():
                optimized_config = self.optimization(df, detector_config, **self.settings["Optimization"][split_type])
                detector_config = optimized_config
        return detector_config

    def get_data(self, save_to):
        """Data ingestion."""
        # get file infos
        root, dirs, files = list(os.walk(self.data_dir))[0]
        n_lines, start_timestamps = self.get_logfiles_info_from_dir()
        files = list(dict(sorted(start_timestamps.items(), key=lambda x: x[1])).keys()) # sort files
        self.input_filepaths = [os.path.join(self.data_dir, file) for file in files]
        # concatenate files and save to tmp folder
        concatenate_files(self.input_filepaths, save_to)
        # get data
        self.df = self.logfile_to_df(path=save_to, interval=None)
        return self.df

    def get_logfiles_info_from_dir(self):
        """Returns number of lines and starting time of log files."""
        root, dirs, files = list(os.walk(self.data_dir))[0]
        n_lines = {}
        # get files sorted by start times of data files
        start_timestamps = {}
        for file in files:
            path = os.path.join(self.data_dir, file)
            with open(path, "r") as f:
                for line in f:
                    start_timestamps[file] = self.get_timestamp_from_string(line) 
                    break # get only first line
            with open(path, "r") as f:
                n_lines[file] = sum(1 for _ in f) # get number of lines for offset
        return n_lines, start_timestamps
    
    def set_and_get_args(self, print_args=True):
        """Configures argument parser and returns command line arguments."""
        parser = argparse.ArgumentParser(description="Automation process for the AMiner.")
        parser.add_argument("-D", "--dataset", type=str, default="russellmitchell", help="Name of dataset.")
        parser.add_argument("-P", "--parser", type=str, default="AuditdParsingModel", help="Type of parser.")
        parser.add_argument("-ts", "--train_splits", type=int, default=1, help="Number of training splits.")
        parser.add_argument("-ms", "--max_training_samples", type=int, default=None, help="Max. number of training samples.")
        parser.add_argument("-id", "--detector_ids", type=str, default="1", help=str(self.detector_id_dict))
        args = parser.parse_args()
        self.__dict__.update(args.__dict__)
        if print_args:
            print(args)

    def init_output_dir(self):
        """Initialize output dir."""
        if self.predefined_config_path!=None:
            prefix = self.predefined_config_path.split(".")[0].split("/")[-1]
        else:
            prefix = "ace"
        self.result_label = f"{prefix}_S{str(len(self.df))}"
        self.output_dir = os.path.join("output", '_'.join(self.detectors), self.parser, self.result_label)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "optimization"), exist_ok=True)