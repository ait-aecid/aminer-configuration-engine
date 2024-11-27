# Configuration-Engine

This code allows to generate configurations for the [logdata-anomaly-miner](https://github.com/ait-aecid/logdata-anomaly-miner) (AMiner) based on static log file analysis.

## **Publication**

The paper "Semi-supervised Configuration and Optimization of Anomaly Detection Algorithms on Log Data" (link will be added after publication) documents the evaluation of the Configuration-Engine.

The evaluation was done with Apache Access and audit log data from [AIT Log Data Set V1.0](https://zenodo.org/records/3723083) and [AIT Log Data Set V2.0](https://zenodo.org/records/5789064). For the individual V2.0 datasets the log files were chosen from <dataset_name>/gather/intranet_server/logs/<data_type>.

## **Installation**

To install the AMiner-Configuration-Engine simply run:
```bash
git clone https://github.com/ait-aecid/aminer-configuration-engine
cd aminer-configuration-engine
git submodule update --init
chmod +x setup.sh
./setup.sh
```


## **Execution**

1. Drop relevant files into directory [data](data). The log data has to be of a single type (e.g. audit or Apache Access). The given sample data in directory [data](data) is Apache Access data from [AIT Log Data Set V2.0](https://zenodo.org/records/5789064) and should be removed before dropping new files. 
2. Execute command (from within the directory):
```bash
python3 create_config.py [-h] [-d DATA_DIR] [-p PARSER_NAME] [-pd USE_PARSED_DATA] [-id DETECTOR_IDS] [-o OPTIMIZE] [-pre PREDEFINED_CONFIG_PATH]
```
For instance, this command will execute the Configuration-Engine with the Apache Access parser for the detectors with IDs 1, 2 and 4 with the optimization turned on.
```bash
python3 create_config.py -d data/ -p ApacheAccessParsingModel -id 1,2,4 -o true
```
For more information:
```bash
python3 create_config.py --help
```

# **For Developers**

## Add new meta-configuration for a detector:

The [meta-configuration](settings/meta-configuration.yaml) file contains the recipes for the detectors' configuration process and the settings for the optimization. The given settings were successfully tested and should be valid for different types of log data. Each detector recipe consists of a composition of one or more **configuration methods**. 

Simply follow the same scheme to define a new meta-configuration for a detector and add it in [meta-configuration.yaml](settings/meta-configuration.yaml) under "ParameterSelection":
```Yaml
# define detector
EntropyDetector:
    # define how the variables (or paths) are selected/filtered
    Variables:
        # pre-filter static variables
        PreFilter:
            Static: {} # "{}" means no additional parameters necessary (because .yaml format)
        # select variables (by character pair probability)
        Select:
            CharacterPairProbability:
                # define specific (hyper) parameter for configuration method
                mean_crit_thresh: 0.7
    # define how specific parameters (of the AMiner config!) are computed
    SpecificParams:
        # choose the method and define its parameters
        CharacterPairProbabilityThresh:
            parameter_name: prob_thresh # name of parameter used in detector
            min: 0.0
            max: 0.9
            offset: -0.05
```
