from src.grouper import Grouper
import numpy as np

class AgglomerativeVariantClustering:
    def __init__(self, grouper: Grouper):
        """
        AgglomerativeVariantClustering

        Parameters
        ------------
        grouper: Grouper; Instance of the used Grouper class (Model Splitting)
        """
        self.grouper: Grouper = grouper
        self.groups_sorted_simplicity: list = sorted(grouper.unique_sorted_place_vec_list, key=lambda x: grouper.statistics[grouper.vec_label_dict[x]]["total_size"], reverse=True)
        self.variants: list = [int(i) for i in grouper.variant_data_raw]
        self.n_variants: int = len(self.variants)
        self.MAX_VAR_ID = max(self.variants)
        self.MAX_N_MODELS: int = 3
        # Global clustering variables
        self.clusterings: list = []
        # Stores for each non-trivial cluster the following info --> group_id: [best group, merged variants]
        self.cluster_info: dict = {}
        # Global selection variable
        self.selections: list = []
        self.selection_scores: list = []
        self.selected_idx: int = -1

    def generate_clusters(self):
        """
        Generate clusters based on the variants and their group vectors.
        """
        # Initialize the clustering variables
        to_merge = self.variants.copy()
        group_idx = max(self.variants) + 1
        # Stores for each non-trivial cluster the following info --> matrix_index: group_id
        grp_row_alloc = {}
        self.clusterings = []
        self.cluster_info = {}

        # Define internal functions for the clustering process
        def get_best_group_for_variants(variants: list[int]) -> tuple[int]:
            # Get the first occurence in groups_sorted_simplicity where all variants are fitting
            best_group = next((vec for vec in self.groups_sorted_simplicity if all(vec[i] == 1 for i in variants)), None)
            return best_group

        def update_matrix(matrix: np.ndarray, i: int, j: int):
            # Set j-th row and column to 0
            matrix[j, :] = 0
            matrix[:, j] = 0
            
            # Update i-th row and column based on the new cluster
            merged_variants = self.cluster_info[grp_row_alloc[i]][1]
            filtered_groups_sorted_simplicity = [vec for vec in self.groups_sorted_simplicity if all(vec[i] == 1 for i in merged_variants)]
            for id in to_merge:
                # Skip the diagonal
                if id == i:
                    continue
                
                # Get the variants that are part of the new cluster
                extended_fitting_variants: set = merged_variants.copy()
                if id in grp_row_alloc:
                    extended_fitting_variants.update(self.cluster_info[grp_row_alloc[id]][1])
                else:
                    extended_fitting_variants.add(id)

                # Get the best group based on the variants in the new cluster and update the matrix
                sup_vec = next((vec for vec in filtered_groups_sorted_simplicity if all(vec[i]==1 for i in extended_fitting_variants)), None)
                simp_val = self.grouper.statistics[self.grouper.vec_label_dict[sup_vec]]["total_size"]
                matrix[i,id] = simp_val
                matrix[id,i] = simp_val
            return

        # Initial distance matrix for simplicity values
        similarity_matrix = np.zeros((self.n_variants, self.n_variants), dtype=int)
        for i in range(self.n_variants):
            for j in range(i+1, self.n_variants):
                # Get the first occurence in groups_sorted_simplicity where the i'th and j'th position is 1
                sup_vec = next((vec for vec in self.groups_sorted_simplicity if vec[i]==1 and vec[j]==1), None)
                simp_val = self.grouper.statistics[self.grouper.vec_label_dict[sup_vec]]["total_size"]
                
                similarity_matrix[i,j] = simp_val
                similarity_matrix[j,i] = simp_val

        while len(to_merge) > 1:
            # Get index i,j of the highest similarity value in the matrix
            min_i, min_j = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
            # min_val = similarity_matrix[min_i, min_j]

            # Get the info which variants were used
            merged_variants = set()
            used_ids = []
            for id in [int(min_i), int(min_j)]:
                # If the id is in grp_row_alloc, then the id does not refer to a variant but to a cluster of variants
                if id not in grp_row_alloc:
                    merged_variants.add(id)
                    to_merge.remove(id)
                    used_ids.append(id)
                else:
                    merged_variants.update(self.cluster_info[grp_row_alloc[id]][1])
                    used_ids.append(grp_row_alloc[id])
                    del grp_row_alloc[id]
                    to_merge.remove(id)

            # Get the best group based on the merged variants
            best_group = get_best_group_for_variants(merged_variants)

            # Store the new cluster info
            self.cluster_info[group_idx] = [best_group, merged_variants]
            grp_row_alloc[min_i] = group_idx
            to_merge.append(min_i)

            # Update the similarity matrix based on the new cluster
            update_matrix(similarity_matrix, min_i, min_j)

            self.clusterings.append([used_ids[0], used_ids[1], self.cluster_info[group_idx][1]])
            # print(f'New Merge {group_idx}: min value {min_val}, {len(self.cluster_info[group_idx][1])}/{sum(best_group)}, fitness {self.grouper.statistics[self.grouper.vec_label_dict[best_group]]['fitness']}, {self.cluster_info[group_idx][1]}')
            group_idx += 1
        return
    
    def generate_selections_from_clusters(self):
        """
        Convert the clusters into selections of group vectors.
        """
        # Define internal function
        def get_groups_after_merge(merge_info: list) -> list:
            # Create a selection based on a list of ids given by the previous clustering
            sel = []
            for j in merge_info:
                # If id j is within the range of variants, then it refers to a variant, else it refers to a cluster
                if j <= self.MAX_VAR_ID:
                    # Check if the group that fulfills only the j-th variant exists and add it if so
                    curr_tuple = [0,] * len(self.groups_sorted_simplicity[0])
                    curr_tuple[j] = 1
                    curr_tuple = tuple(curr_tuple)
                    if curr_tuple in self.groups_sorted_simplicity:
                        sel.append(curr_tuple)
                else:
                    sel.append(self.cluster_info[j][0])
            return sel

        current_merges = set(self.variants)
        initial_selection = get_groups_after_merge(current_merges)

        # Initialize the selection
        if len(initial_selection) == 0:
            self.selections = []
        else:
            self.selections = [initial_selection]

        for i, grp_id in enumerate(self.cluster_info):
            # Remove the ids of the current merge
            current_merges.remove(self.clusterings[i][0])
            current_merges.remove(self.clusterings[i][1])

            # Add the new id of the current merge
            current_merges.add(grp_id)

            # If the next merge is based on the same group as the current, skip to this step
            if (grp_id+1 in self.cluster_info) and (self.cluster_info[grp_id+1][0] == self.cluster_info[grp_id][0]):
                continue
            
            # print(f'Current merge of idx {len(self.selections)}: {self.clusterings[i][2]} -modelFitness-> {self.grouper.statistics[self.grouper.vec_label_dict[self.cluster_info[grp_id][0]]]["fitness"]}')
            # Create the selection and add it
            sel = get_groups_after_merge(current_merges)
            self.selections.append(sel)
        return
    
    def reduce_selections(self):
        """
        Reduce the selections to a maximum number defined by MAX_N_MODELS.
        """
        freq_values = [self.grouper.variant_data_raw[variant]["freq"] for variant in self.grouper.variant_data_raw]
        total_traces = sum(freq_values)

        def get_deletion_score(vec: tuple, selection: list) -> float:
            # Calculate the total fitness of the selection (the variants covered by at least one group)
            total_sel_fitness = sum(self.grouper.variant_data_raw[str(i)]["freq"] * max([x[i] for x in selection]) for i in range(len(selection[0]))) / total_traces
            
            # Remove the considered group from the selection and calculate the total fitness again
            updated_selection = selection.copy()
            updated_selection.remove(vec)
            updated_sel_fitness = sum(self.grouper.variant_data_raw[str(i)]["freq"] * max([x[i] for x in updated_selection]) for i in range(len(updated_selection[0]))) / total_traces
            
            vec_fitness = self.grouper.statistics[self.grouper.vec_label_dict[vec]]['fitness']

            # Unique fitness is what the group contributes to the total fitness (traces that can only be replayed in this vector)
            unique_fitness = total_sel_fitness - updated_sel_fitness
            relative_unique_fitness = min(unique_fitness / vec_fitness, 1.0) # Min is just to avoid some cases where the fitness is so low, that this value ends up being > 1.0 because of rounding errors

            score = 1 - (vec_fitness + relative_unique_fitness)/2
            return round(score, 4)

        # Remove all groups that are not needed
        for sel in self.selections:
            while len(sel) > self.MAX_N_MODELS:
                min_fitness_group = max(sel, key=lambda vec: get_deletion_score(vec, sel))
                sel.remove(min_fitness_group)

        # Reduce duplicated selections after reduction while keeping the order of the selections
        unique_selections = []
        seen = set()
        for selection in self.selections:
            # Convert selection to a tuple for immutability and set operations
            # First, sort the selection based on the groups label (i.e., index) to ensure consistent ordering
            sel = sorted(selection, key=lambda x: self.grouper.vec_label_dict[x])
            selection_tuple = tuple(sel)
            if selection_tuple not in seen:
                seen.add(selection_tuple)
                unique_selections.append(sel)
        self.selections = unique_selections
        return
    
    def get_scored_selections(self):
        """
        Score the selections based on how well each variant is represented in it.
        """
        def get_selection_score(selection: list) -> float:
            if len(selection) == 0:
                return 0.0
            
            variant_coefficients = []
            for variant in range(len(selection[0])):
                # Get first group in groups_sorted_simplicity where the variant is 1
                max_variant_size = self.grouper.statistics[self.grouper.vec_label_dict[next((vec for vec in self.groups_sorted_simplicity if vec[variant] == 1), None)]]["total_size"]
                best_variant_size_sel = max([self.grouper.statistics[self.grouper.vec_label_dict[vec]]["total_size"] for vec in selection if vec[variant] == 1], default=0)
                if best_variant_size_sel != 0:
                    variant_coefficients.append(best_variant_size_sel / max_variant_size)

            score1 = sum(variant_coefficients) / len(selection[0])
            return round(score1,5)

        self.selection_scores = []
        for i,sel in enumerate(self.selections):
            # Calculate the score for each selection
            sel_score = get_selection_score(sel)
            self.selection_scores.append((sel_score, i))

        # Get the index of the selection with the highest score
        self.selected_idx = max(range(len(self.selection_scores)), key=lambda i: self.selection_scores[i][0])
        return
    
    def apply(self, MAX_N_MODELS: int = 3):
        """
        Apply the agglomerative variant clustering algorithm to generate selections of group vectors.
        """
        self.MAX_N_MODELS = MAX_N_MODELS
        self.generate_clusters()
        self.generate_selections_from_clusters()
        self.reduce_selections()
        self.get_scored_selections()
        return