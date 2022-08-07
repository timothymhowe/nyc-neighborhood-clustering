import h3
import pandas as pd
import numpy as np
from h3 import h3_distance
from IPython.display import clear_output
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import nan_euclidean_distances
from copy import copy


class CellFactory:

    def __init__(self, hex_list, data_list, neighborhood_id_list):

        self.hex_list = hex_list
        self.data_list = data_list
        self.neighborhood_id_list = neighborhood_id_list.set_index('hex_id')
        # print(len(self.neighborhood_id_list))

        self.cell_dict = {}
        self.neighborhood_list = []

    # Method for setting up all the cells to be generated
    def generate_cells(self, all_cells=False):
        if all_cells:
            index_list = self.hex_list.join(self.data_list, how="left", lsuffix='L')
        else:
            index_list = self.hex_list.join(self.data_list, how="inner", lsuffix='L')

        for index, row in index_list.iterrows():
            hex_id = row['hex_id']
            # TODO: DON"T HARD CODE THIS, ITS BAD.  Set the correct pre and post conditions for the data dfs!!
            self.cell_dict[hex_id] = HexCell(hex_id, self, data_values=list(row[2:]))

        self.trim_neighbors()

    # creates and populates the list of neighborhoods that will be used to 
    def generate_neighborhoods(self):
        self.neighborhood_list = []
        for hood_id, hood_name in self.neighborhood_id_list.iterrows():
            self.neighborhood_list += [GreedyNeighborhood(self.cell_dict[hood_id], hood_name[0], factory=self)]

    def get_cell_dict(self):
        return self.cell_dict

    def trim_neighbors(self):
        for key in self.cell_dict.keys():
            self.cell_dict[key].neighbor_ids = [the_id for the_id in self.cell_dict[key].neighbor_ids if
                                                the_id in self.cell_dict.keys()]

    def simulate(self, n=2):
        new_hood_list = []
        for i in range(1, n + 1):

            for hood in self.neighborhood_list:
                print(f"Simulating round {i} of {n}")
                print(f"Current Neighborhood: {hood.get_name()}")
                hood.play_turn()
                clear_output(wait=True)
                new_hood_list = new_hood_list + [hood.update_factory()]

        self.neighborhood_list = new_hood_list

    def get_results(self):
        results_list = []
        for hood in self.neighborhood_list:
            # print(hood.output_cell_map())
            results_list = results_list + [hood.output_cell_map()]

        output = pd.concat(results_list)
        return output

    def plot_neighborhoods(self):
        pass

    def do_it_myself(self):
        df = pd.DataFrame()
        for key, cell in self.cell_dict.items():
            df[key] = cell.neighborhood_id

        return df


import random


class HexCell:
    """

    """

    def __init__(self, hex_id, factory, data_values=([None] * cell_len)):
        self.hex_id = hex_id
        self.neighbor_ids = list(h3.k_ring(hex_id))
        self.data_values = data_values
        self.affinity = -1
        self.neighborhood_id = -1
        self.oob = True
        self.factory = factory

    def set_data_values(self, data_values, start_index=0, end_index=cell_len):
        self.data_values = data_values

    # def __copy__(self):
    #     return copy.copy(self)

    def get_data_values(self, fill_nan):
        if fill_nan:
            newvals = []
            for val in self.data_values:
                if pd.isna(val):
                    newvals = newvals + [random.uniform(0, 1)]
                else:
                    newvals = newvals + [val]
            return newvals
        else:
            return self.data_values

    def detect_border(self):
        for id in self.neighbor_ids:
            if self.factory.cell_dict[id].neighborhood_id != self.neighborhood_id:
                return True

        # for id in self.neighbor_ids:
        #           if id in self.factory.cell_dict.keys():
        #               if self.factory.cell_dict[id].neighborhood_id != self.neighborhood_id:
        #                   return True

        return False


# %%
class GreedyNeighborhood:
    """

    """

    def __init__(self, starting_cell, name, factory):
        self.centroid_cell = copy(starting_cell)
        self.centroid_id = self.centroid_cell.hex_id
        self.centroid_cell.neighborhood_id = self.centroid_id

        self.name = name
        self.neighborhood_mean = (self.centroid_cell.get_data_values(True))

        self.factory = factory
        self.factory.cell_dict[self.centroid_id] = self.centroid_cell

        ## initializing lists of cells that are part of the neighborhood
        self.border_cells = [self.centroid_id]
        self.interior_cells = [self.centroid_id]
        self.wanted_cells = []

        # print (self.centroid_cell.neighbor_ids)

    def play_turn(self):
        self.get_wanted_neighbors()
        if len(self.wanted_cells):
            self.claim_cells()

    def resolve_ownership_disputes(self):
        self.add_to_interior([cell for cell in self.get_owned_cells() if
                              self.factory.cell_dict[cell].neighborhood_id == self.centroid_id])

    # gets a list of all neighbors of bordering cells  and then trims any that are already in the neighborhood be
    def get_wanted_neighbors(self):
        for cell_id in self.border_cells:
            cell = self.factory.cell_dict[cell_id]
            potentials = cell.neighbor_ids
            potentials = np.setdiff1d(potentials, self.get_owned_cells())
            # print(potentials)
            # print(self.wanted_cells)
            self.wanted_cells = list(self.wanted_cells) + list(potentials)
            self.wanted_cells = list(set(self.wanted_cells))

    def get_owned_cells(self):
        return self.border_cells + self.interior_cells

    def is_connected(self):
        pass

    # def regression(self):
    #     training_list = [x.data_values for x in self.get_owned_cells()]
    #     regr = LinearRegression()

    def get_distance(self, X, Y, Y_norm_sqd):
        return euclidean_distances(X, Y, Y_norm_squared=Y_norm_sqd, squared=True)

    def get_nan_distance(self, X, Y):
        return nan_euclidean_distances(X, Y, squared=True)

    def claim_cells(self):
        self.neighborhood_mean = self.recompute_mean()
        Y = np.array(self.neighborhood_mean).reshape(1, -1)
        X = np.array([self.factory.cell_dict[cell].get_data_values(True) for cell in self.wanted_cells]).reshape(
            len(self.wanted_cells), -1)
        # X = np.array([self.factory.cell_dict[cell].get_data_values() for cell in self.wanted_cells]).reshape(-1,1)
        # print(f"X.shape, X= {X.shape} , {X}")
        # print(f"Y.shape, y = {(Y.shape)} , {Y}")
        # X = np.array([print(cell) for cell in self.wanted_cells]).reshape(-1,1)

        rez = pd.DataFrame(nan_euclidean_distances(X, Y, squared=True), columns=['dist'])
        # rez = pd.DataFrame(euclidean_distances(X, Y, squared=True), columns=['dist'])

        # Y_norm_sqd = int(np.exp2(Y).sum())
        # rez = pd.DataFrame(euclidean_distances(X, Y, Y_norm_squared=Y_norm_sqd, squared=True), columns=['dist'])

        rez['current_affinity'] = [self.factory.cell_dict[cell].affinity for cell in self.wanted_cells]
        rez['delta_affinity'] = rez['dist'] - rez['current_affinity']
        rez['id'] = self.wanted_cells
        temp = rez[(rez['delta_affinity'] > 0)]

        # print(rez[(rez['delta_affinity'] > 0)])
        claiming_cells = temp['id']
        for _, row in temp.iterrows():
            self.factory.cell_dict[row['id']].affinity = row['dist']
            self.factory.cell_dict[row['id']].neighborhood_id = self.centroid_id

        self.wanted_cells = np.setdiff1d(self.wanted_cells, claiming_cells)
        # self.border_cells = self.border_cells + list(claiming_cells)
        """
        TODO FIX THIS GARBANZO BEANS
        """
        # new_border = [self.factory.cell_dict[cell].detect_border() for cell in self.border_cells]
        # print(new_border)
        # # self.border_cells = self.border_cells[new_border]
        # self.interior_cells += self.interior_cells[new_border == False]

        self.add_to_interior(claiming_cells)

    def determine_border(self):
        new_border = []
        for cell_id in self.border_cells:
            if self.factory.cell_dict[cell_id].neighborhood_id == self.centroid_id:
                new_border = new_border + [cell_id]

    def add_to_interior(self, new_cell_ids):
        check_list = []
        for cell_id in new_cell_ids:
            check_list = check_list + self.factory.cell_dict[cell_id].neighbor_ids

        # check_list  = [x for x in check_list if x in self.factory.cell_dict.keys()]

        to_interior_list = []
        for cell_id in check_list:
            cell = self.factory.cell_dict[cell_id]
            if cell.neighborhood_id == self.centroid_id:
                if not cell.detect_border():
                    to_interior_list = to_interior_list + [cell_id]
                else:
                    self.border_cells = self.border_cells + [cell_id]

        self.interior_cells = list(set(self.interior_cells + to_interior_list))
        self.border_cells = list(set(self.border_cells))

        # TODO: HANDLE BORDER INCURSIONS

    def pick_me(self, neighbor):
        pass

    def agree_on_border(self, neighbor):
        pass

    def recompute_mean(self, nan=False):
        owned_cells = self.get_owned_cells()
        weighted_sum_list = None

        for cell_id in owned_cells:
            cell = self.factory.cell_dict[cell_id]
            weight = 1 / (4 ** h3_distance(self.centroid_id, cell.hex_id))
            component_list = tuple([x * weight for x in cell.get_data_values(True)])
            # print(component_list)
            weighted_sum_list = self.pairwise_tuple_combine(weighted_sum_list, component_list)
        mean = [x / len(owned_cells) for x in weighted_sum_list]
        return mean

    def get_name(self):
        return self.name

    def update_factory(self):
        return self

    def output_cell_map(self):
        out = pd.DataFrame(self.get_owned_cells(), columns=['hex_id'])
        out['name'] = self.name
        out['centroid_id'] = self.centroid_id
        return out

    def output_data(self):
        self.recompute_mean()
        return [self.centroid_id, self.get_owned_cells(), self.neighborhood_mean]

    @staticmethod
    def pairwise_tuple_combine(tup_a, tup_b):
        if tup_a == None:
            # print("Warning: tup_a is 'None'")
            return tup_b
        out = tuple([i + j for i, j in zip(tup_a, tup_b)])
        return out