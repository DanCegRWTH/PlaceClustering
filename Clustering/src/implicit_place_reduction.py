import numpy as np
from scipy.optimize import linprog
from pm4py import PetriNet
from src.utils import get_place_info_from_string, get_unique_transition_list, remove_model_id_str
# This code is the (into python) translated version of the implicit place reduction form the eST-Miner in ProM.

# Check if place is implicit
def is_implicit_among(current_place_index: int, places: list[PetriNet.Place], pre_incidence_matrix: list[list[int]], incidence_matrix: list[list[int]]):
    """
    Check if the given place is implicit among the given places using linear programming (LP).

    Parameters
    ------------
    current_place_index
        Index of the place to check for implicitness
    places
        List of places to check for implicitness
    pre_incidence_matrix
        List of indices of incoming transitions for each place
    incidence_matrix
        List of incidence matrix for each place

    Returns
    -------
    place_inplicit : bool; True if the place is implicit, False otherwise
    """
    assert len(places) == len(pre_incidence_matrix)
    assert len(places) == len(incidence_matrix)
    assert 0 <= current_place_index < len(places)

    place_count = len(places)
    y_variables_count = place_count
    z_variables_count = place_count
    total_variables_count = y_variables_count + z_variables_count + 1 + 1

    k_variable_index = y_variables_count + z_variables_count
    x_variable_index = y_variables_count + z_variables_count + 1

    # Objective function: Minimize k 
    coefficients_linear_objective_function = np.zeros(total_variables_count)
    coefficients_linear_objective_function[k_variable_index] = 1

    # Constraints
    A_eq = []  # For equality constraints (EQ)
    b_eq = []
    A_ub = []  # For less-than-or-equal-to constraints (LEQ)
    b_ub = []
    bounds = [(0, None)] * total_variables_count  # All variables are non-negative

    # Type 0: Ensure current place is not in Y or Z (i.e., currP = 0)
    force_current_place_to_zero_in_y = np.zeros(total_variables_count)
    force_current_place_to_zero_in_y[current_place_index] = 1  # y_{curr_p} = 1
    force_current_place_to_zero_in_z = np.zeros(total_variables_count)
    force_current_place_to_zero_in_z[y_variables_count + current_place_index] = 1  # z_{curr_p} = 1

    # Add equality constraints for currP to be zero in Y and Z
    A_eq.append(force_current_place_to_zero_in_y)
    b_eq.append(0)
    A_eq.append(force_current_place_to_zero_in_z)
    b_eq.append(0)

    # Type 1: Y >= Z >= 0, k >= 0, x = 0, x < k --> x - k <= -1
    for p in range(place_count):
        # Z >= 0 (non-negative constraint on Z)
        non_neg_z = np.zeros(total_variables_count)
        non_neg_z[y_variables_count + p] = 1  # z_p >= 0
        A_ub.append(-non_neg_z) # Flip for >=
        b_ub.append(0)

        # Y >= Z (Y_geq_Z -> y_p - z_p >= 0 -> y_p - z_p <= 0)
        y_geq_z = np.zeros(total_variables_count)
        y_geq_z[p] = 1  # y_p
        y_geq_z[y_variables_count + p] = -1  # -z_p
        A_ub.append(-y_geq_z) # Flip for >=
        b_ub.append(0)

    # k >= 0
    non_neg_k = np.zeros(total_variables_count)
    non_neg_k[k_variable_index] = 1
    A_ub.append(-non_neg_k) # Flip for >=
    b_ub.append(0)

    # x = 0
    zero_x = np.zeros(total_variables_count)
    zero_x[x_variable_index] = 1
    A_eq.append(zero_x)
    b_eq.append(0)

    # x - k <= -1 (x_smaller_k)
    x_smaller_k = np.zeros(total_variables_count)
    x_smaller_k[x_variable_index] = 1  # x
    x_smaller_k[k_variable_index] = -1  # -k
    A_ub.append(x_smaller_k)
    b_ub.append(-1)

    # Type 2: Y * incidence_matrix <= k * incidence_matrix[current_place]
    for t in range(len(incidence_matrix[0])):  # Iterate over transitions
        coefficients = np.zeros(total_variables_count)
        for p in range(place_count):
            coefficients[p] = incidence_matrix[p][t]  # Y part
        coefficients[k_variable_index] = -incidence_matrix[current_place_index][t]  # k part
        A_ub.append(coefficients)
        b_ub.append(0)

    # Type 3: Z * pre_inc_matrix + x >= k * pre(current_place, t)
    pre_inc_transitions = pre_incidence_matrix[current_place_index]
    for encoded_preset_transition in pre_inc_transitions:
        coefficients = np.zeros(total_variables_count)
        for p in range(place_count):
            if encoded_preset_transition in pre_incidence_matrix[p]:
                coefficients[y_variables_count + p] = 1  # Z part
        coefficients[x_variable_index] = 1  # x part
        # Z * pre_inc_matrix + x - k * pre(currP, t) >= 1 -> -Z - x + k <= -1
        #coefficients[k_variable_index] = -pre_incidence_matrix[current_place_index][encoded_preset_transition]
        A_ub.append(-coefficients)  # Multiplying by -1 to convert >= to <=
        b_ub.append(-1)

    # Solve the linear program using SciPy's linprog
    try:
        result = linprog(c=coefficients_linear_objective_function,  # Objective function to minimize (-k)
                         A_ub=np.array(A_ub),  # Inequality constraints (<=)
                         b_ub=np.array(b_ub),
                         A_eq=np.array(A_eq),  # Equality constraints (=)
                         b_eq=np.array(b_eq),
                         bounds=bounds,
                         method='highs')  # Using 'highs' method instead of 'simplex' for stability
        if result.success:
            return True
        else:
            return False
    except ValueError as e:
        # Handle infeasible or unbounded solution
        print(ValueError)
        return False
    

def lp_based_implicit_place_reduction(net: PetriNet):
    """
    Performs implicit place reduction using linear programming (LP) on the given Petri Net.
    Only the places with incoming/outgoing transition info are needed (i.e., of the form "{a,...}|{b,...}").

    Parameters
    ------------
    net
        PetriNet object to perform implicit place reduction on (in-place)
    """
    places = sorted(list(net.places), key=lambda x: len(str(x)))
    survivors = []

    preIncidenceMatrix = []
    incidenceMatrix = []

    unique_transitions = get_unique_transition_list(places, True)

    # Create preIncidenceMatrix (List of indices of incoming transitions per place)
    # Create incidenceMatrix (For each place +1 if transition is incoming, -1 if outgoing, 0 if none or both)
    for place in places:
        place_info = get_place_info_from_string(remove_model_id_str(place.name))
        place_pre_incidence = []
        place_incidence = [0] * len(unique_transitions)
        
        for inc_el in place_info[0]:
            place_incidence[unique_transitions.index(inc_el)] += 1
            place_pre_incidence.append(unique_transitions.index(inc_el))
        for out_el in place_info[1]:
            place_incidence[unique_transitions.index(out_el)] += -1
        
        preIncidenceMatrix.append(place_pre_incidence)
        incidenceMatrix.append(place_incidence)

    n_places = len(places)
    i = 0
    # Check for each place if it is implicit. Delete if so, otherwise keep it.
    while i < n_places:
        if is_implicit_among(i, places, preIncidenceMatrix, incidenceMatrix):
            del places[i]
            del preIncidenceMatrix[i]
            del incidenceMatrix[i]
            n_places -= 1
        else:
            survivors.append(places[i])
            i += 1

    net.places.clear()
    for place in survivors:
        net.places.add(place)
    return