# AMiner-Configuration-Engine

This code allows to generate configurations for the logdata-anomaly-miner (AMiner) based on static log file analysis.


Work in progress ...

## **Execution**

1. Drop relevant files into directory data/. The log data has to be of a single type (e.g., audit or ApacheAccess).
2. Execute command:

```
python3 main.py [-h] [-d DATA_DIR] [-p PARSER] [-pd USE_PARSED_DATA] [-id DETECTOR_IDS] [-o OPTIMIZE] [-pre PREDEFINED_CONFIG_PATH]
```
For instance, this command will execute the Configuration-Engine with the Apache Access parser for the detectors with IDs 1, 2 and 4 with the optimization turned on.
```
python3 main.py -d data/ -p ApacheAccessParsingModel -id 124 -o true
```
For more information:
```
python3 main.py --help
```

## ****

