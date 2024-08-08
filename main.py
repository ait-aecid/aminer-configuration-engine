import time
import argparse
import static_settings
from lib.configUtils import *
from lib.AminerConfigurationEngine import AminerConfigurationEngine as Ace

def get_args():
    """Returns command line arguments."""
    parser = argparse.ArgumentParser(description=
        """AMiner-Configuration-Engine: Drop relevant files (of same log data type) into directory data/ and execute command."""
    )
    parser.add_argument("-d", "--data_dir", type=str, default="/data", help="Directory with data files. All log files in folder will be used as training data.")
    parser.add_argument("-p", "--parser", type=str, default="AuditdParsingModel", help="Type of parser.")
    parser.add_argument("-pd", "--use_parsed_data", type=str, default="true", help="Use already parsed data if data was previsouly parsed? Parsed data is saved temporarily in /tmp.")
    detector_help = f"Choose which detectors to be optimized by their ID (e.g., '13' means detectors with IDs 1 and 3): {str(static_settings.detector_id_dict)}"
    parser.add_argument("-id", "--detector_ids", type=str, default="123456", help=detector_help)
    parser.add_argument("-o", "--optimize", type=str, default="true", help="Optimize detectors?")
    parser.add_argument("-pre", "--predefined_config_path", type=str, default=None, help="Path to a predefined configuration that should be optimized.")
    args = parser.parse_args()
    args.detectors = [static_settings.detector_id_dict[id] for id in args.detector_ids]
    dict_bool = {"true": True, "false": False}
    args.optimize = dict_bool.get(args.optimize.lower(), True)
    args.use_parsed_data = dict_bool.get(args.use_parsed_data.lower(), True)
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
    analysis_config = ace.configure_detectors(ace.predefined_config)
    config_runtime = time.time()-start
    print(f"Configuration completed (runtime: {config_runtime}).\n")

    # add trivial config parts
    ace.config["LearnMode"] = True
    ace.config["LogResourceList"] = [os.path.join(ace.current_dir, path) for path in ace.input_filepaths]
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