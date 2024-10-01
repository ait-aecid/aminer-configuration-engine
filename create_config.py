import time
import argparse
import os
from settings.constants import DETECTOR_ID_DICT
from lib.utils import *
from lib.ConfigurationEngine import ConfigurationEngine

HELP_MESSAGE = """                                                                                                                               
   (                   (                                          )                                                                     
   )\                  )\ )   (    (  (      (    (        )   ( /(   (                        (             (  (    (              (   
 (((_)    (     (     (()/(   )\   )\))(    ))\   )(    ( /(   )\())  )\    (     (            )\     (      )\))(   )\    (       ))\  
 )\___    )\    )\ )   /(_)) ((_) ((_))\   /((_) (()\   )(_)) (_))/  ((_)   )\    )\ )        ((_)    )\ )  ((_))\  ((_)   )\ )   /((_) 
((/ __|  ((_)  _(_/(  (_) _|  (_)  (()(_) (_))(   ((_) ((_)_  | |_    (_)  ((_)  _(_/(   ___  | __|  _(_/(   (()(_)  (_)  _(_/(  (_))   
 | (__  / _ \ | ' \))  |  _|  | | / _` |  | || | | '_| / _` | |  _|   | | / _ \ | ' \)) |___| | _|  | ' \)) / _` |   | | | ' \)) / -_)  
  \___| \___/ |_||_|   |_|    |_| \__, |   \_,_| |_|   \__,_|  \__|   |_| \___/ |_||_|        |___| |_||_|  \__, |   |_| |_||_|  \___|  
                                  |___/                                                                     |___/                       

AMiner-Configuration-Engine: Drop relevant files (of same log data type) into directory data/ and execute command."""

def get_args():
    """Returns command line arguments."""
    parser = argparse.ArgumentParser(description=HELP_MESSAGE, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--data_dir", type=str, default="/data", help="Directory with data files. All log files in folder will be used as training data.")
    parser.add_argument("-p", "--parser", type=str, default="AuditdParsingModel", help="Type of parser.")
    parser.add_argument("-pd", "--use_parsed_data", type=str, default="true", help="Use already parsed data if data was previsouly parsed? Parsed data is saved temporarily in /tmp.")
    detector_help = f"Choose which detectors to be optimized by their ID (e.g., '13' means detectors with IDs 1 and 3): {str(DETECTOR_ID_DICT)}"
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
    """Main function of the configuration automation process for the AMiner."""
    
    # initialize AminerConfigurationEngine
    Ce = ConfigurationEngine(params)

    # run configuration methods
    print("\nConfiguring detectors ...")
    start = time.time()
    analysis_config = Ce.configure_detectors(Ce.predefined_config)
    config_runtime = time.time()-start
    print(f"Configuration completed (runtime: {config_runtime}).\n")

    # add necessary parts to config
    Ce.config["LearnMode"] = True
    Ce.config["LogResourceList"] = [os.path.join(Ce.current_dir, path) for path in Ce.input_filepaths]
    Ce.config["Analysis"] = analysis_config

    # save config
    config_path = os.path.join(Ce.output_dir, "config.yaml")
    dump_config(config_path, Ce.config)
    print("Configuration file saved to:", config_path)


if __name__ == "__main__":
    input_args = get_args()
    print(input_args)
    main(input_args)