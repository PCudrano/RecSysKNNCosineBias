#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix

from Data_manager.DataReaderPostprocessing import DataReaderPostprocessing
from Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens, removeFeatures, remove_empty_rows_and_cols
import numpy as np



class DataReaderPostprocessing_K_Cores(DataReaderPostprocessing):
    """
    This class selects a dense partition of URM such that all items and users have at least K interactions.
    The algorithm is recursive and might not converge until the graph is empty.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """


    def __init__(self, dataReader_object, k_cores_value):

        assert k_cores_value >= 1,\
            "DataReaderPostprocessing_K_Cores: k_cores_value must be a positive value >= 1, provided value was {}".format(k_cores_value)

        super(DataReaderPostprocessing_K_Cores, self).__init__(dataReader_object)

        self.k_cores_value = k_cores_value



    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "{}_cores/".format(self.k_cores_value)

        inner_subfolder_name = self.dataReader_object._get_dataset_name_data_subfolder()

        # Avoid concatenating the original/ part
        if inner_subfolder_name != self.DATASET_SUBFOLDER_ORIGINAL:
            subfolder_name += inner_subfolder_name

        return subfolder_name



    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the k-cores
        :return:
        """

        self.dataReader_object.load_data()

        self.URM_all = self.dataReader_object.get_URM_all()
        self.item_original_ID_to_index = self.dataReader_object.item_original_ID_to_index
        self.user_original_ID_to_index = self.dataReader_object.user_original_ID_to_index

        n_users, n_items = self.URM_all.shape

        # Apply required K - core on zero-core data from ORIGINAL split
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        print("DataReaderPostprocessing_K_Cores: Removed {} users and {} items with less than {} interactions".format(len(removedUsers), len(removedItems), self.k_cores_value))

        ICM_filter_mask = np.ones(n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False


        print("DataReaderPostprocessing_K_Cores: Removing items from ICMs... ")


        loaded_ICMs = self.dataReader_object.get_loaded_ICM_names()

        for ICM_name in loaded_ICMs:

            print("DataReaderPostprocessing_K_Cores: Removing items from {}".format(ICM_name))

            ICM_object = self.dataReader_object.get_ICM_from_name(ICM_name)
            ICM_object = ICM_object[ICM_filter_mask,:]

            ICM_mapper_name = "tokenToFeatureMapper_{}".format(ICM_name)
            ICM_mapper_object = self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name)

            ICM_object, _, ICM_mapper_object = removeFeatures(ICM_object, minOccurrence = 1, maxPercOccurrence = 1.00,
                                                                reconcile_mapper = ICM_mapper_object)

            setattr(self, ICM_name, ICM_object)
            setattr(self, ICM_mapper_name, ICM_mapper_object)



        print("DataReaderPostprocessing_K_Cores: Removing items from ICMs... done")







def select_k_cores(URM, k_value = 5, reshape = False):
    """

    :param URM:
    :param k_value:
    :param reshape:
    :return: URM, removedUsers, removedItems
    """

    print("DataDenseSplit_K_Cores: k-cores extraction will zero out some users and items without changing URM shape")

    URM.eliminate_zeros()

    n_users = URM.shape[0]
    n_items = URM.shape[1]

    removed_users = set()
    removed_items = set()

    print("DataDenseSplit_K_Cores: Initial URM desity is {:.2E}".format(URM.nnz/(n_users*n_items)))

    convergence = False
    numIterations = 0

    while not convergence:

        convergence_user = False

        URM = check_matrix(URM, 'csr')

        user_degree = np.ediff1d(URM.indptr)

        to_be_removed = user_degree < k_value
        to_be_removed[np.array(list(removed_users), dtype=np.int)] = False

        if not np.any(to_be_removed):
            convergence_user = True

        else:

            for user in range(n_users):

                if to_be_removed[user] and user not in removed_users:
                    URM.data[URM.indptr[user]:URM.indptr[user+1]] = 0
                    removed_users.add(user)

            URM.eliminate_zeros()



        convergence_item = False

        URM = check_matrix(URM, 'csc')

        items_degree = np.ediff1d(URM.indptr)

        to_be_removed = items_degree < k_value
        to_be_removed[np.array(list(removed_items), dtype=np.int)] = False

        if not np.any(to_be_removed):
            convergence_item = True

        else:

            for item in range(n_items):

                if to_be_removed[item] and item not in removed_items:
                    URM.data[URM.indptr[item]:URM.indptr[item+1]] = 0
                    removed_items.add(item)

            URM.eliminate_zeros()




        numIterations += 1
        convergence = convergence_item and convergence_user


        if URM.data.sum() == 0:
            convergence = True
            print("DataDenseSplit_K_Cores: WARNING on iteration {}. URM is empty.".format(numIterations))

        else:
             print("DataDenseSplit_K_Cores: Iteration {}. URM desity without zeroed-out nodes is {:.2E}.\n"
                  "Users with less than {} interactions are {} ( {:.2f}%), Items are {} ( {:.2f}%)".format(
                numIterations,
                sum(URM.data)/((n_users-len(removed_users))*(n_items-len(removed_items))),
                k_value,
                len(removed_users), len(removed_users)/n_users*100,
                len(removed_items), len(removed_items)/n_items*100))


    print("DataDenseSplit_K_Cores: split complete")

    URM.eliminate_zeros()

    if reshape:
        # Remove all columns and rows with no interactions
        return remove_empty_rows_and_cols(URM)


    return URM.copy(), np.array(list(removed_users)), np.array(list(removed_items))
