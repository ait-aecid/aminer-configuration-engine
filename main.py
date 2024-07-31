import time
from lib.configUtils import *
from lib.AminerConfigurationEngine import AminerConfigurationEngine as Ace

def main(
    label="",
    predefined_config=None, 
):
    """Main function of the configuration automation process for the AMiner."""

    params = {
        "detectors" : ["NewMatchPathValueDetector"],
        "data_dir" : "/home/viktor/projects/final/aminer-configuration-engine/data/data_split/russellmitchell/gather/intranet_server/logs/audit/",
        "parser" : "AuditdParsingModel",
        "detector_ids" : "123456",
        "predefined_config_path" : None,
    }
    # initialize AminerConfigurationEngine
    ace = Ace(params)

    # run configuration methods
    print("\nConfiguring detectors ...")
    start = time.time()
    analysis_config = ace.configure_detectors(predefined_config)
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
    #input_args = get_args()
    #main(*input_args)
    main()