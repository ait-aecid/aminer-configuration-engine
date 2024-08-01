import time
import argparse
from lib.configUtils import *
from lib.AminerConfigurationEngine import AminerConfigurationEngine as Ace

def get_args():
    """Configures argument parser and returns command line arguments."""
    detector_id_dict = {
        "1" : "NewMatchPathValueDetector",
        "2" : "NewMatchPathValueComboDetector",
        "3" : "CharsetDetector",
        "4" : "EntropyDetector",
        "5" : "ValueRangeDetector",
        "6" : "EventFrequencyDetector"
    }
    parser = argparse.ArgumentParser(description="Automation process for the AMiner.")
    parser.add_argument("-p", "--parser", type=str, default="AuditdParsingModel", help="Type of parser.")
    parser.add_argument("-d", "--data_dir", type=str, default="/data", help="Directory with data files. All log files in folder will be used as training data.")
    parser.add_argument("-id", "--detector_ids", type=str, default="123456", help=str(detector_id_dict))
    parser.add_argument("-pre", "--predefined_config_path", type=str, default=None, help="Path to a predefined configuration that should be optimized.")
    parser.add_argument("-o", "--optimize", type=str, default="true", help="")
    args = parser.parse_args()
    args.detectors = [detector_id_dict[id] for id in args.detector_ids]
    dict_bool = {"true": True, "false": False}
    args.optimize = dict_bool.get(args.optimize.lower(), True)
    return args.__dict__

def main(
    params,
    label="",
):
    """Main function of the configuration automation process for the AMiner."""
    # initialize AminerConfigurationEngine
    ace = Ace(params)

    # run configuration methods
    print("\nConfiguring detectors ...")
    start = time.time()
    analysis_config = ace.configure_detectors(params["predefined_config_path"])
    config_runtime = time.time()-start
    print(f"Configuration completed (runtime: {config_runtime})\n")

    # add trivial config parts
    ace.config["LearnMode"] = True
    ace.config["LogResourceList"] = ace.input_filepaths
    ace.config["Analysis"] = analysis_config

    # save config
    config_path = os.path.join(ace.output_dir, "config.yaml")
    dump_config(config_path, ace.config)
    print("Configuration file saved to:", config_path)

    return config_runtime

if __name__ == "__main__":
    input_args = get_args()
    print(input_args)
    main(input_args)