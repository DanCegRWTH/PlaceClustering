from src.grouper import Grouper
import pm4py as pm
from IPython.display import display, clear_output
import ipywidgets as widgets
from src.create_petrinet import create_petri_net_from_place_list, merge_petri_nets
from src.utils import get_place_info_from_string, get_unique_transition_list
from src.model_selection import AgglomerativeVariantClustering
import time

class Visualizer:
    def __init__(self, place_file_path: str):
        """
        Visualizer class handles the creation of Petri nets, their merging based on a heuristic, and their visualization.

        Parameters
        ------------
        place_file_path: str; Path to the place file
        """
        # Definition of instance variables
        # Grouper related variables
        self.grouper: Grouper = None
        self.interesting_groups: list = []
        # Visualizer related variables
        self.model_list: list = []
        self.selected_index: int = -1
        self.net_id: int = 0
        # Output widgets that are automatically updated
        self.out_models = widgets.Output()
        self.out_slider = widgets.Output()

        st_time = time.time()
        self.grouper = Grouper(place_file_path)
        et_time = time.time()
        print(f"Model Splitting took {et_time - st_time:.2f} seconds...")
        self.interesting_groups = [] 

        self.model_cache = {}

        self.model_selection = AgglomerativeVariantClustering(self.grouper)

        self.update_out_models()

    def filter_place_list(self, places: list[str], not_list: list[str]):
        """
        Filter out places that contain activities (incoming and outgoing arcs) from the not_list.

        Parameters
        ------------
        places : List[str]; List of places to filter
        not_list : List[str]; List of activities to filter out

        Returns
        -------
        filtered_places : List[str]; List of filtered places
        """
        # If not_list is empty, return the original list
        if not_list == []:
            return places
        # If not_list is not empty, only return places that do not contain incoming or outgoing arcs from the not_list
        filtered_places = []
        for place in places:
            place_info = get_place_info_from_string(place)
            if all(char not in place_info[0] and char not in place_info[1] for char in not_list):
                filtered_places.append(place)
        return filtered_places

    def get_filtered_place_list(self, groups_to_use: list[int], activities_to_use: list[str]):
        """
        Get list of places to use for the Petri net creation. The places are filtered based on the transitions that are unique to the group in groups_to_use.

        Parameters
        ------------
        groups_to_use : List[int]; List of group labels (index) to use for the Petri net creation
        activities_to_use : List[str]; List of activities to use for the Petri net creation

        Returns
        -------
        filtered_places : List[str]; List of filtered places to use for the Petri net creation
        """
        # get all unique transitions (these are contained in the perfect places)
        all_unique_transitions = get_unique_transition_list(self.grouper.perfect_places)
        # Get list of transitions to filter out
        not_list = list(set(all_unique_transitions) - set(activities_to_use))
        
        filtered_places = []
        # For a list of groups like [0,1]
        for group in groups_to_use:
            filtered_places += self.filter_place_list(self.grouper.places_by_group[group], not_list)
        # Add filtered perfect places as they are considered as a separate group and not in places_by_group
        #filtered_places += self.filter_place_list(self.grouper.perfect_places, not_list) NOT NEEDED
        return filtered_places

    def update_out_models(self):
        """
        Update the output widget that displays the Petri nets.
        """
        with self.out_models:
            clear_output(wait=True)
            for i, model in enumerate(self.model_list):
                print(f"Model ID: {i}")
                pm.view_petri_net(model[0], model[1], model[2])

    def get_slider_output(self):
        """
        Update the output widget that displays the selection slider for model selection.
        """
        with self.out_slider:
            clear_output(wait=True)
            if not self.model_selection.selections:
                print("No selections yet. Please run the model selection first.")
                return
            
            slider_options = [sel_info[1] for sel_info in self.model_selection.selection_scores]
            # Get index of selection_scores where the selection score is the highest
            best_sel_val = self.model_selection.selection_scores[self.selected_index][1]
            
            slider = widgets.SelectionSlider(
                options=slider_options,
                value=best_sel_val,
                description='Similarity:',
                disabled=False,
                continuous_update=False,
                readout=True
            )

            def on_slider_change(change):
                self.selected_index = change.owner.index

            slider.observe(on_slider_change, names='value')
            # Display the sliders
            display(slider)
        display(self.out_slider)

    def update_selection(self):
        """
        Update the displayed models based on the current selection index.
        """
        new_selection = self.model_selection.selections[self.selected_index]
        self.set_interesting_groups(new_selection)

    def set_interesting_groups(self, interesting_vecs: list):
        """
        Set the displayed models based on the provided group vectors.

        Parameters
        ------------
        interesting_vecs : List[int]; List of group vectors to display
        """
        # Get the groups needed to create the models of the interesting vectors
        self.interesting_groups = [self.grouper.group_collections[self.grouper.vec_label_dict[i]] for i in interesting_vecs]
        # Reset the model list and net_id
        self.model_list = []
        self.net_id = 0
        # Creat Petri net for each interesting group
        for group in self.interesting_groups:
            # Cache the models to avoid recomputing them (address with the group label)
            if group[0] not in self.model_cache:
                net, im, fm = create_petri_net_from_place_list(self.get_filtered_place_list(group, self.grouper.activities_by_group[group[0]]), self.net_id, True)
                self.model_cache[group[0]] = (net, im, fm)
            self.model_list.append(self.model_cache[group[0]])
            self.net_id += 1
        
        self.update_out_models()
    
    def get_models_output(self):
        """
        Call this function to display the models
        """
        display(self.out_models)

    def merge_models_in_list(self):
        """
        Merge the models in the model list into a single Petri net and update the model list.
        They are merged by combining the source places into one source place and the target places into one target place.
        """
        new_net, new_im, new_fm = merge_petri_nets(self.model_list)
        self.net_id = 0
        self.model_list = [(new_net, new_im, new_fm)]

        self.update_out_models()

    def apply_model_selection(self, MAX_N_MODELS: int = 3):
        """
        Apply the model selection algorithm to the current models and update the interesting groups.

        Parameters
        ------------
        MAX_N_MODELS : int; Maximum number of models in the selection (default is 3)
        """
        st_time = time.time()
        self.model_selection.apply(MAX_N_MODELS)
        et_time = time.time()
        print(f"Model Selection took {et_time - st_time:.2f} seconds...")
        highest_index = max(range(len(self.model_selection.selection_scores)), key=lambda i: self.model_selection.selection_scores[i][0])
        self.selected_index = highest_index
        best_sel = self.model_selection.selections[highest_index]
        print(f'Creating Petri nets...')
        self.set_interesting_groups(best_sel)