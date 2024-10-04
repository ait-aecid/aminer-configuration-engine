
import os
import pandas as pd
import sys
import time
import importlib

# custom imports
from lib.utils import *
from lib.Optimization import Optimization

# import from submodule
import sys
sys.path.append('log-preprocessor')
from Data import Data
from utils.constants import *

class ConfigurationEngine(Optimization):
    """This class contains all the functionality that is required for the initialization of this project."""

    def __init__(self, params: dict):
        """Initialize project. Returns parsed command line arguments."""

        self.__dict__.update(params)
        self.detector_id_dict = DETECTOR_ID_DICT
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
        # load base config
        self.config = load_yaml_file("settings/base_config.yml")
        self.config["Parser"][0]["type"] = self.parser_name
        with open("settings/meta-configuration.yaml", 'r') as yaml_file:
            self.settings = yaml.safe_load(yaml_file)

        # get the data
        start = time.time()
        data = Data(
            self.data_dir,
            self.parser_name,
            self.config["Input"]["timestamp_paths"],
            tpm_save_path=self.data_path
        )
        self.df = data.get_df()
        print(f"Finished data extraction (runtime: {time.time() - start}).")
        self.input_filepaths = data.input_filepaths

        self.init_output_dir()
        self.current_dir = "file://" + os.getcwd()

    def configure_detectors(self, predefined_config=None) -> list:
        """Configure detectors and return their configurations as dictionaries in a list."""
        df = self.df.copy()
        detector_config = []
        if predefined_config == None:
            for current_detector in self.detectors:
                if current_detector != "EventFrequencyDetector": # TO-DO: should be handled in meta-config
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

    def init_output_dir(self):
        """Initialize output directory."""
        if self.predefined_config_path!=None:
            prefix = self.predefined_config_path.split(".")[0].split("/")[-1]
        else:
            prefix = "CE"
        self.result_label = f"{prefix}_S{str(len(self.df))}"
        self.output_dir = os.path.join("output", '_'.join(self.detectors), self.parser_name, self.result_label)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "optimization"), exist_ok=True)