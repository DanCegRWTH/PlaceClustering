from pm4py import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
# Own imports
from src.implicit_place_reduction import lp_based_implicit_place_reduction
from src.utils import get_place_info_from_string, get_unique_transition_list, remove_model_id_str, extract_model_id_str

def add_arcs_of_place(net: PetriNet, place: PetriNet.Place, label_transition_dict: dict = {}):
    """
    Add arcs to the Petri Net based on the place string "{a}|{b}" (i.e., icoming arc from a and outgoing arc to b).

    Parameters
    ------------
    net
        PetriNet object to add arcs to (in-place)
    place
        Place object to add arcs for with label format "{a,...}|{c,...}"
    label_transition_dict
        Dictionary mapping transition label to transition object
    """
    place_info = get_place_info_from_string(remove_model_id_str(place.name))
    for inc_el in place_info[0]:
        petri_utils.add_arc_from_to(label_transition_dict[inc_el], place, net)
    
    for out_el in place_info[1]:
        petri_utils.add_arc_from_to(place, label_transition_dict[out_el], net)
    return

def add_src_snk_im_fm(src_str: str, snk_str: str, net: PetriNet, label_transition_dict: dict = {}):
    """
    Add source and sink places to the Petri Net and return the initial and final markings.

    Parameters
    ------------
    src_str
        String of the source place label of the form "{}|{▷}"
    snk_str
        String of the sink place label of the form "{☐}|{}"
    net
        PetriNet object to add source and sink places to (in-place)
    label_transition_dict
        Dictionary mapping transition label to transition object

    Returns
    -------
    im : initial marking of the PetriNet (type: Marking)
    fm : final marking of the PetriNet (type: Marking)
    """
    source_place = PetriNet.Place(src_str)
    net.places.add(source_place)
    add_arcs_of_place(net, source_place, label_transition_dict)

    sink_place = PetriNet.Place(snk_str)
    net.places.add(sink_place)
    add_arcs_of_place(net, sink_place, label_transition_dict)

    im = Marking({source_place: 1})
    fm = Marking({sink_place: 1})
    return im, fm

def add_all_arcs(net: PetriNet, label_transition_dict: dict):
    """
    Add all arcs to the Petri Net based on its places (of the form "{a}|{b}"). If thre are existing arcs, they are cleared.

    Parameters
    ------------
    net
        PetriNet object to add arcs to (in-place)
    label_transition_dict
        Dictionary mapping transition label to transition object
    """
    net.arcs.clear()
    temp_places = list(net.places)
    for place in temp_places:
        add_arcs_of_place(net, place, label_transition_dict)
    return

def reduce_self_loop_places(net: PetriNet, label_transition_dict: dict) -> None:
    """
    Reduce the number of places containing self-loop. Merge places like {a,b}|{b,c} and {a,d}|{d,c} to {a,b,d}|{b,d,c}. The net is modified in-place.

    Parameters
    ------------
    net
        PetriNet object to add arcs to (in-place)
    label_transition_dict
        Dictionary mapping transition label to transition object
    """
    net_places = net.places.copy()
    place_arc_dict = {}
    for place in net_places:
        place_info = get_place_info_from_string(remove_model_id_str(place.name))
        # First check if it contains a self-loop (i.e., element is in both input and output)
        self_loop_labels = set([el for el in place_info[0] if el in place_info[1]])

        if not self_loop_labels:
            continue  # Skip if no self-loop labels are found
        
        # If we are still here, then the place contains a self-loop
        # Get all corresponding arcs and split the transitions into incoming and outgoing
        place_arcs = list(filter(lambda x: x.source == place or x.target == place, net.arcs))
        inc_arcs_from = [inc_arc.source for inc_arc in place_arcs if inc_arc.target == place]
        out_arcs_to = [out_arc.target for out_arc in place_arcs if out_arc.source == place]

        # For each self-loop, remove the transitions from the infos
        for label in self_loop_labels:
            inc_arcs_from.remove(label_transition_dict[label])
            out_arcs_to.remove(label_transition_dict[label])

        new_place = True
        # Check if there is another place with similar input and output labels (except the self-loop)
        for other_place in place_arc_dict:
            # If the incoming arcs are not the same, we skip (len of both lists must be equal to prevent inc_arcs being a subset)
            if not (all(inc_el in place_arc_dict[other_place]['inc_arcs'] for inc_el in inc_arcs_from) and len(inc_arcs_from) == len(place_arc_dict[other_place]['inc_arcs'])):
                continue
            # Similar to incoming arcs, we check the outgoing arcs
            if not (all(out_el in place_arc_dict[other_place]['out_arcs'] for out_el in out_arcs_to) and len(out_arcs_to) == len(place_arc_dict[other_place]['out_arcs'])):
                continue
            # If we are still here, then the place is similar to another place
            new_place = False
            place_arc_dict[other_place]['new_self_loop_labels'].update(self_loop_labels)
            place_arc_dict[other_place]['merged'] = True

            # Remove the current place and the corresponding arcs from the net
            net.places.remove(place)
            for arc in place_arcs:
                net.arcs.remove(arc)
            break

        # If no similar place was found, we add the new place to the dictionary
        if new_place:
            place_arc_dict[place] = {
                'inc_arcs': inc_arcs_from,
                'out_arcs': out_arcs_to,
                'new_self_loop_labels': set(),
                'merged': False
            }

    for place, info in place_arc_dict.items():
        if not info['merged']:
            continue

        for label in info['new_self_loop_labels']:
            petri_utils.add_arc_from_to(label_transition_dict[label], place, net)
            petri_utils.add_arc_from_to(place, label_transition_dict[label], net)
    return

def create_petri_net_from_place_list(place_list: list[str], net_id: int = 0, implcit_place_redux = True):
    """
    Create a Petri Net from list of place strings (of the form "{a,...}|{b,...}"). The transitions are inferred from the place strings.

    Parameters
    ------------
    place_list
        List of places of the form "{a,...}|{b,...}"
    implcit_place_redux
        Bool. If True, implicit place reduction is applied. Default is True.

    Returns
    ------------
    net : PetriNet object (pm4py)
    im : initial marking of the PetriNet (type: Marking)
    fm : final marking of the PetriNet (type: Marking)
    """
    net = PetriNet(f'model_split_net_{net_id}')

    unique_transitions = get_unique_transition_list(place_list)

    label_transition_dict = {}

    for tran_el in unique_transitions:
        tran_name = tran_el + f'_{net_id}'
        label_transition_dict[tran_el] = PetriNet.Transition(tran_name, tran_el)
        net.transitions.add(label_transition_dict[tran_el])

    for place_str in place_list:
        place = PetriNet.Place(place_str + f'_{net_id}')
        net.places.add(place)

    if implcit_place_redux:
        lp_based_implicit_place_reduction(net)
    
    add_all_arcs(net, label_transition_dict)
    im,fm = add_src_snk_im_fm("{}|{▷}" + f'_{net_id}', "{☐}|{}" + f'_{net_id}', net, label_transition_dict)

    if implcit_place_redux:
        # Reduce self-loop places: merge {a,b}|{b,c} and {a,d}|{d,c} to {a,b,d}|{b,d,c}
        reduce_self_loop_places(net, label_transition_dict)

    return net, im, fm

def merge_petri_nets(model_list: list[tuple[PetriNet, Marking, Marking]], start_activity: str = '▷', end_activity: str = '☐') -> tuple[PetriNet, Marking, Marking]:
    """
    Merge multiple Petri nets into one. The source places are merged into one source place and the sink places are merged into one sink place.

    Parameters
    ------------
    model_list
        List of tuples containing PetriNet, initial marking and final marking
    start_activity
        String of the start activity label (default is '▷')
    end_activity
        String of the end activity label (default is '☐')

    Returns
    ------------
    net : PetriNet
        Merged Petri net
    im : Marking
        Initial marking of the merged Petri net
    fm : Marking
        Final marking of the merged Petri net
    """
    # Creating the new Petri net by merging the places, transitions, and arcs, caused pm4py errors when executing conformance checking.
    new_net = PetriNet("Combined Model")

    model_start_places = []
    model_end_places = []
    # Iterate over the models in model_list and add their transitions and places to the new net
    for net, im, fm in model_list:
        place_list = [remove_model_id_str(place.name) for place in net.places]
        unique_transitions = get_unique_transition_list(place_list)

        label_transition_dict = {}
        net_id = extract_model_id_str(list(net.places)[0].name)
        # Add transitions to the new net
        for tran_el in unique_transitions:
            tran_name = tran_el + f'_{net_id}'
            label_transition_dict[tran_el] = PetriNet.Transition(tran_name, tran_el)
            new_net.transitions.add(label_transition_dict[tran_el])
            
            # Mark the start and end transitions to link them to the new source and sink places
            if tran_el == start_activity:
                model_start_places.append(label_transition_dict[tran_el])
            elif tran_el == end_activity:
                model_end_places.append(label_transition_dict[tran_el])

        new_places = []
        # Add places to the new net
        for place in net.places:
            # Skip the source and the sink place of the original net
            if place == list(im)[0] or place == list(fm)[0]:
                continue
            place = PetriNet.Place(place.name)
            new_net.places.add(place)
            new_places.append(place)

        # Add arcs to the new net
        for place in new_places:
            add_arcs_of_place(new_net, place, label_transition_dict)

    # Add the new merged source place
    new_source_place = PetriNet.Place("newSource")
    new_net.places.add(new_source_place)
    for start_tran in model_start_places:
        petri_utils.add_arc_from_to(new_source_place, start_tran, new_net)
    new_im = Marking({new_source_place: 1})

    # Add the new merged sink place
    new_end_place = PetriNet.Place("newEnd")
    new_net.places.add(new_end_place)
    for end_tran in model_end_places:
        petri_utils.add_arc_from_to(end_tran, new_end_place, new_net)
    new_fm = Marking({new_end_place: 1})

    return new_net, new_im, new_fm