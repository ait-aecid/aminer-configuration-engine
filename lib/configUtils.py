from ruamel import yaml
from lib.ParameterSelection import *

def flatten_dict(d: dict) -> list:
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(flatten_dict(value))
        elif isinstance(value, list):
            values.extend(value)
    return values

def load_yaml_file(file_path):
    """Load .yaml file into a dictionary."""
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error while loading YAML file: {e}")

def dump_config(filename : str, configuration : dict):
    """Create config file from dictionary."""
    with open(filename, "w") as file:
        yaml.dump(configuration, file, sort_keys=False, indent=4)

def assemble_detector(detector: str, params: dict) -> list:
    """Assemble the configuration of a detector."""

    instance_list = []
    used_variables = set()
    filter_list = []
    # if "Filter" in params["Variables"].keys():
    #     filter_list = flatten_dict(params["Variables"]["PostFilter"])
    i = 0
    for method in params["Variables"]["Select"].keys():
        for variable in params["Variables"]["Select"][method]:
            if type(variable) == list:
                paths = variable
                variable = tuple(variable)
            else:
                paths = [variable]
            # skip duplicates and filtered variables
            if (variable in used_variables) or (variable in filter_list):
                continue
            else:
                used_variables.add(variable)

            # exception for EFD
            paths_str = "paths"
            if detector == "EventFrequencyDetector":
                paths_str = "constraint_list"

            # define instance
            identifier = f"{detector}_{method}_id{i}"
            instance = {
                "type": detector,
                "id" : identifier,
                "persistence_id" : identifier,
                paths_str: paths,
                "output_logline": True,
            }
            # remove 'paths' if the variable is 'None'
            if pd.isna(variable):
                instance.pop(paths_str, None)
            # add specific parameters
            if "SpecificParams" in params.keys():
                specific_params = {}
                for s_method, name_dict in params["SpecificParams"].items():
                    for name, param_dict in name_dict.items():
                        specific_params[name] = param_dict[variable]
                instance.update(specific_params)
            instance_list.append(instance)
            i += 1
    return instance_list

def adapt_predefined_analysis_config(analysis_config, detectors, df, print_deleted=False):
    """Adapt a predefined analysis config by filtering instances that were not specified or contain variables that are not given in the data."""

    allowed_items=["type","id","paths","persistence_id","output_logline", "season", "num_windows", "confidence_factor", "window_size", "prob_thresh"]
    adapted_config = []
    deleted_items = {"types": [], "paths": []}
    remaining_types = []
    remaining_paths = []

    conf = analysis_config.copy()
    for item in conf:
        if item["type"] not in detectors:
            deleted_items["types"].append(item["type"])
        elif not any(path in df.columns for path in item["paths"]):
            deleted_items["paths"].extend(item["paths"])
        else:
            item["output_logline"] = True
            new_item = {key: val for key, val in item.items() if key in allowed_items}
            remaining_types.append(new_item["type"])
            remaining_paths.append(new_item["paths"])
            adapted_config.append(new_item)
    if print_deleted:
        print(f"Remaining types: {remaining_types}")
        print(f"Remaining paths: {remaining_paths}")
    return adapted_config