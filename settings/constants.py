# If a new configuration method for a new detector is defined, the detector name has to be mapped to a (single) character.
DETECTOR_ID_DICT = {
    "1" : "NewMatchPathValueDetector",
    "2" : "NewMatchPathValueComboDetector",
    "3" : "CharsetDetector",
    "4" : "EntropyDetector",
    "5" : "ValueRangeDetector",
    "6" : "EventFrequencyDetector",
    "7" : "EventSequenceDetector"
}

# Define how the timestamps should be extracted for each parser. A new parser type requires a new definition.
TIMESTAMP_EXTRACTION_DICT = {
    'AuditdParsingModel': {
        'timestamp_format': None,
        'split_char': ["msg=audit(", ":"],
        "unit" : "s",
    },
    'ApacheAccessParsingModel': {
        'timestamp_format': '%d/%b/%Y:%H:%M:%S %z',
        'split_char': ["[", "]"],
        "unit": None
    }
}

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