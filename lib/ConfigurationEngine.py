
import os
import pandas as pd
import sys
import time
import importlib

from settings.constants import DETECTOR_ID_DICT, TIMESTAMP_EXTRACTION_DICT
from lib.utils import *
from lib.Optimization import Optimization

sys.path.append("/usr/lib/logdata-anomaly-miner")
sys.path.append("/etc/aminer/conf-available/ait-lds")
from aminer.parsing.MatchElement import MatchElement
from aminer.parsing.MatchContext import MatchContext
from aminer.input.LogAtom import LogAtom
from aminer.parsing.ParserMatch import ParserMatch

class ConfigurationEngine(Optimization):
    """This class contains all the functionality that is required for the initialization of this project."""

    def __init__(self, params: dict):
        """Initialize project. Returns parsed command line arguments."""

        self.__dict__.update(params)
        self.detector_id_dict = DETECTOR_ID_DICT
        self.file_type_info = TIMESTAMP_EXTRACTION_DICT
        self.label = "TBA"
        self.predefined_config = None

        # create tmp directory
        os.makedirs("tmp", exist_ok=True)
        os.makedirs(os.path.join("tmp", "data_parsed"), exist_ok=True)
        self.data_path = os.path.join("tmp", "current_data.log")
        # if a predefined config was passed load it and just optimize it - see optimization
        if self.predefined_config_path != None:
            self.predefined_config = load_yaml_file(self.predefined_config_path)
        # get the detector names from their ids
        self.detectors = [self.detector_id_dict[id] for id in self.detector_ids.split(",")]
        # get the data
        start = time.time()
        self.df = self.get_data(save_to=self.data_path)
        print(f"Finished data extraction (runtime: {time.time() - start}).")
        self.init_output_dir()
        self.current_dir = "file://" + os.getcwd()
        # define standard inputs and fill config
        self.config = load_yaml_file("settings/base_config.yml")
        self.config["Parser"][0]["type"] = self.parser
        with open("settings/meta-configuration.yaml", 'r') as yaml_file:
            self.settings = yaml.safe_load(yaml_file)

    def logfile_to_df(self, path: str, interval=None):
        """Get a list of match dictionaries from log data."""
        module = importlib.import_module(self.parser)
        parsing_model = module.get_model()
        # similar to process in /usr/lib/logdata-anomaly-miner/aminer/input/ByteStreamLineAtomizer.py
        match_dict_list = []
        timestamps = []
        # for faster repeated data ingestion
        h5_label = "-".join([p.split("/")[-1] for p in self.input_filepaths])
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
    
    def get_timestamp_from_string(self, string: str):
        """Get timestamp from string."""
        split_char = self.file_type_info[self.parser]["split_char"]
        ts_string = string.split(split_char[0])[1].split(split_char[1])[0]
        ts = pd.to_datetime(ts_string, format=self.file_type_info[self.parser]["timestamp_format"], unit=self.file_type_info[self.parser]["unit"])
        return ts

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
                instance["persistence_id"] = f"instance_id_{i}"
                detector_config[i] = instance
            if "id" not in instance.keys():
                instance["id"] = f"instance_id_{i}"
                detector_config[i] = instance
        # optimize
        if len(detector_config) > 0 and self.optimize:
            for split_type in self.settings["Optimization"].keys():
                opt_settings = self.settings["Optimization"][split_type]
                optimized_config = self.optimization(df, detector_config, **opt_settings)
                if len(optimized_config) == 0:
                    print("Optimization leads to empty configuration. Original config. is used.")
                else: 
                    detector_config = optimized_config
        return detector_config

    def get_data(self, save_to: str):
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

    def init_output_dir(self):
        """Initialize output directory."""
        if self.predefined_config_path!=None:
            prefix = self.predefined_config_path.split(".")[0].split("/")[-1]
        else:
            prefix = "ace"
        self.result_label = f"{prefix}_S{str(len(self.df))}"
        self.output_dir = os.path.join("output", '_'.join(self.detectors), self.parser, self.result_label)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "optimization"), exist_ok=True)