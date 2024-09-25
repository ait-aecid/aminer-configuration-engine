# AMiner-Configuration-Engine

This code allows to generate configurations for the [logdata-anomaly-miner](https://github.com/ait-aecid/logdata-anomaly-miner) (AMiner) based on static log file analysis.

**Optimization is not yet working! (requires adaptations in AMiner - coming soon ...)**

## **Installation**


## **Execution**

1. Drop relevant files into directory [data](data). The log data has to be of a single type (e.g. audit or ApacheAccess). The given sample data in [data](data) is from [AIT Log Data Set V2.0](https://zenodo.org/records/5789064) and should be removed before dropping new files. The 
2. Execute command:

```bash
python3 main.py [-h] [-d DATA_DIR] [-p PARSER] [-pd USE_PARSED_DATA] [-id DETECTOR_IDS] [-o OPTIMIZE] [-pre PREDEFINED_CONFIG_PATH]
```
For instance, this command will execute the Configuration-Engine with the Apache Access parser for the detectors with IDs 1, 2 and 4 with the optimization turned on.
```bash
python3 main.py -d data/ -p ApacheAccessParsingModel -id 124 -o true
```
For more information:
```bash
python3 main.py --help
```

# **For Developers**

## Add new meta-configuration for a detector:

The [meta-configuration](meta-configuration.yaml) file contains the recipes for the detectors' configuration process and the settings for the optimization. The given settings were successfully tested and should be valid for different types of log data. Each detector recipe consists of a composition of one or more **configuration methods**. 

Simply follow the same scheme to define a new meta-configuration for a detector and add it in [meta-configuration.yaml](meta-configuration.yaml) under "ParameterSelection":
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

## Add new configuration methods:

TBA

