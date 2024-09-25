# If a new configuration method for a new detector is defined, the detector name has to be mapped to a (single) character.
detector_id_dict = {
            "1" : "NewMatchPathValueDetector",
            "2" : "NewMatchPathValueComboDetector",
            "3" : "CharsetDetector",
            "4" : "EntropyDetector",
            "5" : "ValueRangeDetector",
            "6" : "EventFrequencyDetector",
            "7" : "EventSequenceDetector"
        }

# Define how the timestamps should be extracted for each parser. A new parser type requires a new definition.
timestamp_extraction_dict = {
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
