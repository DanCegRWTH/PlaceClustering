from src.grouper import Grouper
import numpy as np
import pandas as pd

### Petri Net Utils
def get_place_info_from_string(place_str: str):
    """
    Convert a string like "{a, b, ▷}|{a, b, ☐}" to a list of lists like [['a', 'b', '▷'], ['a', 'b', '☐']] 
    where the first list is incoming transitions and the second list is outgoing transitions.
    (Potential error if activities have , or | in them.)

    Parameters
    ------------
    place_str
        String to convert of the form "{a, b, ▷}|{a, b, ☐}"

    Returns
    -------
    place_info : List; place_info[0] - list of incoming transitions, place_info[1] - list of outgoing transitions
    """
    # Remove {, }, and spaces, then split by | to get two parts
    parts = place_str.replace('{', '').replace('}', '').split('|')
    # Split each part by , to get the list of transitions
    result = [part.split(', ') for part in parts]
    # Convert [''] to []
    result = [part if part != [''] else [] for part in result]
    return result

def get_unique_transition_list(place_list: list, contains_net_id: bool = False):
    """
    Returns a list of unique transition labels from a list of places (of the form "{a,...}|{b,...}").

    Parameters
    ------------
    place_list
        list of places with labels or strings of the form "{a,...}|{b,...}"

    Returns
    -------
    unique_transitions_list : list of unique transition labels
    """
    unique_transitions = set()
    for place_str in place_list:
        place_info = get_place_info_from_string(remove_model_id_str(str(place_str))) if contains_net_id else get_place_info_from_string(place_str)
        
        unique_transitions.update(place_info[0])
        unique_transitions.update(place_info[1])
    return list(unique_transitions)

def get_costs(grouper: Grouper, list_of_selections: list):
    n_variants = grouper.N_VARIANTS
    total_traces = grouper.N_TOTAL_TRACES
    unique_groups = grouper.unique_sorted_place_vec_list

    s_1_values = [round(1 - 1/(1+np.e**(-1.2*(i-4))), 5) for i in range(1, 5)]
    variant_freqs = np.array([grouper.variant_data_raw[str(i)]["freq"] for i in range(n_variants)])
    vec_fitness = np.array([grouper.statistics[grouper.vec_label_dict[x]]["fitness"] for x in unique_groups])

    print("Calculating costs...")
    costs = np.zeros((len(list_of_selections), 3))
    for idx, selection in enumerate(list_of_selections):
        if idx % 1000000 == 0:
            print(f"Calculating costs for selection {idx}...")
        selection_indices = [unique_groups.index(tuple(x)) for x in selection]
        s_1 = s_1_values[len(selection)-1]
        s_2 = round(1 - vec_fitness[selection_indices].max(), 5)
        s_3 = round((variant_freqs * np.max([unique_groups[i] for i in selection_indices], axis=0)).sum() / total_traces, 5)
        costs[idx] = [s_1, s_2, s_3]
    
    return costs

def add_unique_start_end(df: pd.DataFrame, unique_start = '▷', unique_end = '☐', case_col='case:concept:name', activity_col='concept:name', timestamp_col='time:timestamp'):
    """
    Adds a unique start and end activity to each trace in the event log (DataFrame).
    The start activity is added one second before the first event of each trace, and the end activity is added one second after the last event of each trace.

    Parameters
    ------------
    df : DataFrame
        DataFrame containing the event log with columns for case ID, activity, and timestamp.
    unique_start : str
        Unique start activity label to be added to the start of each trace (default is '▷').
    unique_end : str
        Unique end activity label to be added to the end of each trace (default is '☐').
    case_col : str
        Name of the column containing case IDs (default is 'case:concept:name').
    activity_col : str
        Name of the column containing activity names (default is 'concept:name').
    timestamp_col : str
        Name of the column containing timestamps (default is 'time:timestamp').

    Returns
    -------
    df : DataFrame
        DataFrame with unique start and end activities added to each trace.
    """
    # Sort the DataFrame by case_id and timestamp
    df = df.sort_values(by=[case_col, timestamp_col])

    # Create a DataFrame for the unique start activities
    start_df = df.groupby(case_col).first().reset_index()
    start_df[timestamp_col] = start_df[timestamp_col] - pd.Timedelta(seconds=1)
    start_df[activity_col] = unique_start
    
    # Create a DataFrame for the unique end activities
    end_df = df.groupby(case_col).last().reset_index()
    end_df[timestamp_col] = end_df[timestamp_col] + pd.Timedelta(seconds=1)
    end_df[activity_col] = unique_end
    
    # Concatenate the start, original, and end DataFrames
    df = pd.concat([start_df, df, end_df], ignore_index=True)
    
    # Sort the DataFrame by case_id and timestamp again
    df = df.sort_values(by=[case_col, timestamp_col]).reset_index(drop=True)
    return df

# Extract the int at the end of a string
def extract_model_id_str(s: str):
    return int(s.split('_')[-1]) if '_' in s else None

# Get the string without the _int
def remove_model_id_str(s: str):
    return '_'.join(s.split('_')[:-1]) if '_' in s else s