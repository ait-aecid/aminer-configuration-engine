import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")

from lib.methods.simple_methods import Static, Random, MinMaxOccurence, Selection
from lib.methods.Stability import Stability
from lib.methods.CharacterPairProbability import CharacterPairProbability
from lib.methods.CoOccurenceCombos import CoOccurrenceCombos
from lib.methods.EventFrequencyAnalysis import EventFrequencyAnalysis

class ParameterSelection(
    Static, 
    Random,
    MinMaxOccurence,
    Selection,
    Stability, 
    CharacterPairProbability, 
    CoOccurrenceCombos,
    EventFrequencyAnalysis,
):
    """This class handles the selection of parameters by mapping the meta-configuration to the configuration methods."""

    def __init__(self, df, settings={}):       
        self.df_original = df.copy()
        self.df = df.copy()

        # map meta-parameters to functions
        mappings = {
            "Variables" : {
                "Selection" : self.get_selected_variables,
                "MinMaxOccurrence" : self.get_variables_by_occurrence,
                "Static" : self.get_static_variables,
                "Random" : self.get_random_variables,
                "Stable" : self.get_stable_variables,
                "CharacterPairProbability" : self.get_variables_by_charPairProb,
                "CoOccurrenceCombos" : self.get_combos_by_co_occurrence,
                "EventFrequencyAnalysis" : self.event_frequency_analysis,
            },
            "SpecificParams" : {
                "CharacterPairProbabilityThresh" : self.get_charPairProb_thresh,
                "EventFrequencyParams": self.get_event_frequency_params,
                "EventSequenceLength": self.get_event_sequence_length,
            }
        }
        # if not empty map meta-parameters to functions
        if settings:
            params = settings.copy()
            self.selected_vars = []
            v = "Variables"
            for action in ["PreFilter", "PreSelect", "Select"]: #TO-DO: implement "PostFilter"
                if action in settings[v].keys():
                    for method in settings[v][action].keys():
                        variables = mappings[v][method](**settings[v][action][method])
                        params[v][action][method] = variables
                        if action == "PreFilter":
                            self.df = self.df.drop(columns=variables)
                        elif action == "PreSelect":
                            flattened_vars = [i for sublist in variables for i in sublist] if type(variables[0]) == list else variables
                            self.df = self.df[flattened_vars]
                            
            s = "SpecificParams"
            if s in settings.keys():
                for method in settings[s].keys():
                    specific_param = mappings[s][method](**settings[s][method])
                    params[s][method] = specific_param
            self.params = params