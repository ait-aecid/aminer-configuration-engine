import json
import numpy as np
import pandas as pd
import shutil
import itertools

# unused rn
def get_results_file(filename='/tmp/aminer_out.json', save=False, save_to="tmp/aminer_out.json"):
    """Get output file that was generated from running the AMiner. JSON format is expected ('pretty=false')."""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    if save:
        shutil.copy(filename, save_to)
    return data

def get_alert_idx_from_file(filename='/tmp/aminer_out.json', offset1=0, offset2=0, save=False, save_to="tmp/aminer_out.json"):
        """Returns the row numbers that triggered alerts."""
        offset2 = offset2 + 1 # offset is exclusive
        alerts = set()
        with open(filename, "r") as file:
            for line in file:
                alert_rel = json.loads(line)["LineNumber"]
                if offset1 == 0:
                    alert = alert_rel + offset2
                else:
                    if alert_rel < offset1:
                        alert = alert_rel
                    else:
                        alert = alert_rel + offset2 - offset1
                alerts.add(alert)
        if save:
            shutil.copy(filename, save_to)
        return list(alerts)
        
def get_relevant_info(detectors) -> dict:
    """Returns detector type, id of the triggered instance, line index, timestamp and variable(s) for each alert."""
    results = get_results_file()
    info = []
    for detector in detectors:
        for i in range(len(results)):
            if results[i]['AnalysisComponent']["AnalysisComponentType"].startswith(detector):
                var = results[i]['AnalysisComponent']["AffectedLogAtomPaths"]
                idx = results[i]["LineNumber"]
                ts = pd.to_datetime(results[i]['LogData']["Timestamps"], unit="s")
                crit = results[i]['AnalysisComponent']["CriticalValue"] if "CriticalValue" in results[i]['AnalysisComponent'].keys() else None
                id = results[i]['AnalysisComponent']["AnalysisComponentName"]
                info.append({"detector":detector, "id":id, "var":var, "idx":idx, "ts":ts, "crit":crit})
    return pd.DataFrame(info, columns=["detector", "id", "var", "idx", "ts", "crit"])


    

