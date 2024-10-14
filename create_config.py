import time
import argparse
import os
from settings.constants import HELP_MESSAGE
from lib.utils import *
from lib.ConfigurationEngine import ConfigurationEngine

# import from submodule
import sys
sys.path.append('log-preproconfiguratorssor')
from utils.constants import *

def get_args():
    """Returns command line arguments."""
    parser = argparse.ArgumentParser(description=HELP_MESSAGE, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--data_dir", type=str, default="/data", help="Directory with data files. All log files in folder will be used as training data.")
    parser.add_argument("-p", "--parser_name", type=str, default="AuditdParsingModel", help="Type of parser.")
    parser.add_argument("-pd", "--use_parsed_data", type=str, default="true", help="Use already parsed data if same data was previsouly parsed? Parsed data is saved temporarily in /tmp.")
    detector_help = f"Choose which detectors to be optimized by their ID (e.g., '1,3' means detectors with IDs 1 and 3): {str(DETECTOR_ID_DICT)}"
    parser.add_argument("-id", "--detector_ids", type=str, default="1,2,3,4,5,6,7", help=detector_help)
    parser.add_argument("-o", "--optimize", type=str, default="true", help="Optimize detectors?")
    parser.add_argument("-pre", "--predefined_config_path", type=str, default=None, help="Path to a predefined configuration that should be optimized.")
    args = parser.parse_args()
    args.detectors = [DETECTOR_ID_DICT[id] for id in args.detector_ids.split(",")]
    dict_bool = {"true": True, "false": False}
    args.optimize = dict_bool.get(args.optimize.lower(), True)
    args.use_parsed_data = dict_bool.get(args.use_parsed_data.lower(), True)
    return args.__dict__

def main(params):
    """Main function of the configuration automation proconfiguratorss for the AMiner."""
    
    # initialize AminerConfigurationEngine
    configurator = ConfigurationEngine(params)

    # run configuration methods
    print("\nConfiguring detectors ...")
    start = time.time()
    analysis_config = configurator.configure_detectors(configurator.predefined_config)
    config_runtime = time.time()-start
    print(f"Configuration completed (runtime: {config_runtime}).\n")

    # add neconfiguratorssary parts to config
    configurator.config["LearnMode"] = True
    configurator.config["LogResourconfiguratorList"] = [os.path.join(configurator.current_dir, path) for path in configurator.input_filepaths]
    configurator.config["Analysis"] = analysis_config

    # save config
    config_path = os.path.join(configurator.output_dir, "config.yaml")
    dump_config(config_path, configurator.config)
    print("Configuration file saved to:", config_path)


if __name__ == "__main__":
    input_args = get_args()
    print(input_args)
    main(input_args)