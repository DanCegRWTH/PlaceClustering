import json
from collections import Counter

class Grouper:
    # Class variables, shared between instances: None
    
    def __init__(self, place_file_path):
        """
        The Grouper class takes as input a JSON file containing place information and variant information. It then groups the places based on the vector representation of a place 
        (1 if the place fits the variant, 0 if not) and calculates a lot of useful information for further use.

        Parameters
        ------------
        place_file_path : str; Path to the place file
        """
        # Definition of instance variables
        # Raw data variables
        self.place_data_raw: dict = None
        self.variant_data_raw: dict = None
        # Place list variables
        self.total_place_list: list[str] = None
        self.perfect_places: list[str] = None
        # Group variables
        self.place_group_labels: list[int] = []
        self.vec_label_dict: dict = {}
        self.places_by_group: list[list] = []
        self.group_collections: list[list] = []
        self.activities_by_group: list[list] = []
        self.groups_superset_dict: dict = {}
        # Place vector variables
        self.place_vec_list: list[tuple[int]] = None
        self.unique_place_vec_list: list[tuple[int]] = None
        self.unique_sorted_place_vec_list: list[tuple[int]] = None
        # Further variables
        self.N_VARIANTS: int = None
        self.N_TOTAL_TRACES: int = None
        self.statistics: dict = {}

        def create_vector_list():
            """
            Convert the place data into a replayability vector for each place.
            """
            # Create a list where each place is represented by a vector with 1 if the place fits the variant and 0 if not
            self.place_vec_list = []

            for place in self.total_place_list:
                indices: str = self.place_data_raw[place]
                indices = indices.strip('{}').split(', ')

                place_vec = [0] * self.N_VARIANTS
                for idx in indices:
                    place_vec[int(idx)] = 1
                place_vec = tuple(place_vec)
                self.place_vec_list.append(place_vec)

        def group_places():
            """
            Assign each place a label.
            """
            # Group places based on the same vector representation
            group_label = 0
            for vec in self.place_vec_list:
                if vec not in self.vec_label_dict:
                    self.vec_label_dict[vec] = group_label
                    group_label += 1
                self.place_group_labels.append(self.vec_label_dict[vec])

        def get_group_stats():
            """
            Create stats for each group.
            """
            # Get the total number of traces, the number of label occurrences, and the groups sorted by label
            total_traces = sum(self.variant_data_raw[str(i)]["freq"] for i in range(self.N_VARIANTS))
            label_counter = Counter(self.place_group_labels)
            groups_sorted_label = list(self.vec_label_dict)
            groups_sorted_label.sort(key=lambda x: self.vec_label_dict[x])

            # Create statistics for the formed groups
            max_group_label = max(self.place_group_labels)
            for i in range(max_group_label+1):
                # Get all place vectors in group i (should be only one unique vector multiple times, but I want to future-proof this)
                self.statistics[i] = {}
                self.statistics[i]["size"] = label_counter[i]
                self.statistics[i]["group_vec"] = groups_sorted_label[i]
                self.statistics[i]["fitness"] = round(sum(self.statistics[i]["group_vec"][k]*self.variant_data_raw[str(k)]["freq"] for k in range(self.N_VARIANTS))/total_traces, 5)

        def get_places_by_group():
            """
            Create a list of lists containing the places for each group.
            """
            # Group the place strings by the group labels and store it in places_by_group
            self.places_by_group = [[] for _ in range(max(self.place_group_labels)+1)]
            for j, place in enumerate(self.total_place_list):
                self.places_by_group[self.place_group_labels[j]].append(place)

        def form_group_collections():
            """
            Generate collections of groups based on the grouping data.
            For each group, add other group to the collection iff each tuple element is less than or equal to the other group's tuple element.
            """
            for vec in self.unique_place_vec_list:
                # Add current group label to collection
                group_col = [self.vec_label_dict[other_vec] for other_vec in self.groups_superset_dict[vec]]
                group_col.insert(0, self.vec_label_dict[vec])
                self.group_collections.append(group_col)

                # Get unique activities for each group
                curr_act_set = set()
                for i, fit in enumerate(vec):
                    # If tuple has a 1 at index i, this group fulfills this variant and we add the activities to the set
                    if fit:
                        curr_variant = self.variant_data_raw[str(i)]['var']
                        curr_act_set = curr_act_set | set(curr_variant.strip("[]").split(", "))
                self.activities_by_group.append(list(curr_act_set))
        
        # Load the JSON file
        with open(place_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Split the data into place and variant info
        self.place_data_raw = data['place_info']
        self.variant_data_raw = data['variant_info']

        self.N_VARIANTS = max(len(a.split(",")) for a in self.place_data_raw.values())
        self.N_TOTAL_TRACES = sum(self.variant_data_raw[str(i)]["freq"] for i in range(self.N_VARIANTS))
        # Create place lists
        TOTAL_PLACE_LIST = list(self.place_data_raw.keys())
        self.total_place_list = list(filter(lambda x: self.place_data_raw[x]!='{}', TOTAL_PLACE_LIST))
        self.perfect_places = list(filter(lambda x: len(self.place_data_raw[x].split(","))==self.N_VARIANTS, TOTAL_PLACE_LIST))

        # Create a list of vector representations for each place
        create_vector_list()
        # Group the places
        group_places()
        # Get statistics for each group
        get_group_stats()
        # Set unique vector lists. Sort the unique vectors by the number of fitting traces
        self.unique_place_vec_list = list(self.vec_label_dict.keys())
        self.unique_sorted_place_vec_list = sorted(self.unique_place_vec_list, key=lambda x: self.statistics[self.vec_label_dict[x]]["fitness"])
        # Get places by group
        get_places_by_group()
        # Precompute the group supersets for each group prevent duplicate calculations
        self.groups_superset_dict = self.precompute_group_supersets()
        # Form group collections
        form_group_collections()

        # Add additional information to statistics
        for i in self.statistics:
            total_size = 0
            for j in self.group_collections[i]:
                total_size += self.statistics[j]['size']
            self.statistics[i]["total_size"] = total_size

    def precompute_group_supersets(self):
        """
        Precompute the supersets of each group in this grouper instance.
        """
        # Function to convert a replayability vector like (0,1,0,1) into a bitmask represented by an int (5 in this case)
        def tuple_to_bitmask(vec: tuple[int]) -> int:
            return sum(value << idx for idx, value in enumerate(reversed(vec)))
        
        # A list of unique vectors sorted by the number of fitting traces (ascending)
        sorted_unique_groups = sorted(self.unique_sorted_place_vec_list, key=lambda x: sum(x))
        bitmask_list = [tuple_to_bitmask(t) for t in sorted_unique_groups]

        # Since this is a computationally expensive operation (when there are many groups), we try to make it as efficient as possible by using bitmasks.
        superset_dict = {}
        n_groups = len(sorted_unique_groups)
        for i in range(n_groups):
            bm_vec = bitmask_list[i]
            supersets = []
            for j in range(i + 1, n_groups):
                bm_candidate = bitmask_list[j]
                # A group is a superset of a considered group if the bitmask of the considered group remains unchanged when using bitwise AND
                if (bm_candidate & bm_vec) == bm_vec:
                    supersets.append(sorted_unique_groups[j])
            superset_dict[sorted_unique_groups[i]] = supersets
        return superset_dict