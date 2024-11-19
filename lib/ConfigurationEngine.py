
import os
import pandas as pd
import sys
import time
import time
from typing import Optional

# custom imports
from lib.utils import *
from lib.Optimization import Optimization

# import from submodule
import sys
sys.path.append('log-preprocessor')
from LogData import LogData
from utils.constants import DETECTOR_ID_DICT

class ConfigurationEngine(Optimization):
    """This class contains all the functionality that is required for the initialization of this project."""

    def __init__(
            self,
            data_dir: str,
            parser_name: str,
            detector_ids="1,2,3,4,5,6,7",
            optimize=True,
            predefined_config_path: Optional[str]=None,
            use_parsed_data=True,
            tmp_save_path="/tmp/current_data.log",
        ):
        """Initialize project. Returns parsed command line arguments."""
        self.data_dir = data_dir
        self.parser_name = parser_name
        self.optimize = optimize
        self.predefined_config_path = predefined_config_path
        self.tmp_save_path = tmp_save_path
        self.detector_ids = detector_ids
        self.detectors = [DETECTOR_ID_DICT[id] for id in detector_ids.split(",")]
        # get the data
        print("\n------------------------- DATA EXTRACTION -------------------------")
        start = time.time()
        data = LogData(
            self.data_dir,
            self.parser_name,
            tmp_save_path=self.tmp_save_path
        )
        self.df = data.get_df(use_parsed_data)
        self.input_filepaths = data.input_filepaths
        print(f"Data extraction finished. (runtime: {round(time.time() - start, 3)}s)")        
        self.init_output_dir()

        # load base config
        self.config = load_yaml_file("settings/base_config.yml")
        self.config["Parser"][0]["type"] = self.parser_name
        with open("settings/meta-configuration.yaml", 'r') as yaml_file:
            self.settings = yaml.safe_load(yaml_file)

        # if a predefined config was passed load it and just optimize it - see optimization
        self.predefined_config = None
        if self.predefined_config_path != None:
            self.predefined_config = load_yaml_file(self.predefined_config_path)

    def configure_detectors(self, predefined_config=None, print_progress=True) -> list:
        """Configure detectors and return their configurations as dictionaries in a list."""
        if print_progress:
            print("\n-------------------------- CONFIGURATION --------------------------")
        start = time.time()
        df = self.df.copy()
        analysis_config = []
        if predefined_config == None:
            for current_detector in self.detectors:
                if print_progress:
                    print(f"Configuring {current_detector} ...")
                if current_detector != "EventFrequencyDetector": # TO-DO: should be handled in meta-config
                    current_df = df.drop(columns="ts")
                else:
                    current_df = df
                current_analysis = ParameterSelection(current_df, self.settings["ParameterSelection"][current_detector])
                analysis_config += assemble_detector(current_detector, current_analysis.params)
        else:
            analysis_config = adapt_predefined_analysis_config(predefined_config["Analysis"], self.detectors, df, print_deleted=True)
        # give a id if no id is given
        for i, instance in enumerate(analysis_config):
            if "persistence_id" not in instance.keys():
                instance["persistence_id"] = f"instance_id_{i}"
                analysis_config[i] = instance
            if "id" not in instance.keys():
                instance["id"] = f"instance_id_{i}"
                analysis_config[i] = instance
        # optimize
        if print_progress:
            print(f"Configuration finished. (runtime: {round(time.time() - start, 3)}s)")
        start = time.time()
        if len(analysis_config) > 0 and self.optimize:
            if print_progress:
                print("\n-------------------------- OPTIMIZATION ---------------------------")
            for split_type in self.settings["Optimization"].keys():
                opt_settings = self.settings["Optimization"][split_type]
                optimized_config = self.optimize_config(df, analysis_config, **opt_settings)
                if len(optimized_config) == 0:
                    print("Optimization leads to empty configuration. Original config. is used.")
                else: 
                    analysis_config = optimized_config
                if print_progress:
                    print(f"Optimization finished. (runtime: {round(time.time() - start, 3)}s)")
        return analysis_config

    def init_output_dir(self):
        """Initialize output directory."""
        if self.predefined_config_path!=None:
            prefix = self.predefined_config_path.split("/")[-1]
        else:
            prefix = "CE"
        self.result_label = f"{prefix}_{str(len(self.df))}_samples"
        output_dir_rel = os.path.join("output", "ids_" + self.detector_ids.replace(",", "-"), self.parser_name, self.result_label)
        self.output_dir = os.path.abspath(output_dir_rel)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "optimization"), exist_ok=True)
    
    def create_config(self):
        """Create the configuration file."""
        analysis_config = self.configure_detectors(self.predefined_config)
        # add necessary parts to config
        self.config["LearnMode"] = True
        self.config["LogResourceList"] = [os.path.join("file://" + os.getcwd(), path) for path in self.input_filepaths]
        self.config["Analysis"] = analysis_config
        # save config
        config_path = os.path.join(self.output_dir, "config.yaml")
        dump_config(config_path, self.config)
        print("\nConfiguration file saved to:", config_path)
        return self.config