import time
import argparse
import os
from settings.constants import HELP_MESSAGE
from lib.utils import *
from lib.ConfigurationEngine import ConfigurationEngine

# import from submodule
import sys
sys.path.append('log-preprocessor')
from utils.constants import *

def get_args():
    """Returns command line arguments."""
    detector_help = f"Choose which detectors to be optimized by their ID (e.g., '1,3' means detectors with IDs 1 and 3): {str(DETECTOR_ID_DICT)}"
    parser = argparse.ArgumentParser(description=HELP_MESSAGE, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--data_dir", type=str, default="/data", help="Directory with data files. All log files in folder will be used as training data.")
    parser.add_argument("-p", "--parser_name", type=str, default="ApacheAccessParsingModel", help="Type of parser.")
    parser.add_argument("-pd", "--use_parsed_data", type=str, default="true", help="Use already parsed data if same data was previsouly parsed? Parsed data is saved temporarily in /tmp.")
    parser.add_argument("-id", "--detector_ids", type=str, default="1,2,3,4,5,6,7", help=detector_help)
    parser.add_argument("-o", "--optimize", type=str, default="true", help="Optimize detectors?")
    parser.add_argument("-pre", "--predefined_config_path", type=str, default=None, help="Path to a predefined configuration that should be optimized.")
    args = parser.parse_args()
    dict_bool = {"true": True, "false": False}
    args.optimize = dict_bool.get(args.optimize.lower(), True)
    args.use_parsed_data = dict_bool.get(args.use_parsed_data.lower(), True)
    return args.__dict__

def main(params):
    """Main function of the configuration automation process for the AMiner."""
    configurator = ConfigurationEngine(**params)
    configurator.create_config()

if __name__ == "__main__":
    input_args = get_args()
    print(input_args)
    main(input_args)