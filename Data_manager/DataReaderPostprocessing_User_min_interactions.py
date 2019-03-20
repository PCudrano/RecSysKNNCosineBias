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
import scipy.sparse as sps



class DataReaderPostprocessing_User_min_interactions(DataReaderPostprocessing):
    """
    This class selects a partition of URM such that all users have at least min_interactions.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """


    def __init__(self, dataReader_object, min_interactions):

        assert min_interactions >= 1,\
            "DataReaderPostprocessing_User_min_interactions: min_interactions must be a positive value >= 1, provided value was {}".format(min_interactions)

        super(DataReaderPostprocessing_User_min_interactions, self).__init__(dataReader_object)

        self.min_interactions = min_interactions



    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "{}_user_min_interactions/".format(self.min_interactions)

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

        # Apply required min user interactions ORIGINAL split
        self.URM_all, removedUsers, removedItems = select_users_with_min_interactions(self.URM_all, min_interactions = self.min_interactions, reshape=True)

        print("DataReaderPostprocessing_User_min_interactions: Reconciling mappers with removed tokens")

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        print("DataReaderPostprocessing_User_min_interactions: Removed {} users with less than {} interactions, and {} items".format(len(removedUsers), self.min_interactions, len(removedItems)))

        ICM_filter_mask = np.ones(n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False


        print("DataReaderPostprocessing_User_min_interactions: Removing items from ICMs... ")


        loaded_ICMs = self.dataReader_object.get_loaded_ICM_names()

        for ICM_name in loaded_ICMs:

            print("DataReaderPostprocessing_User_min_interactions: Removing items from {}".format(ICM_name))

            ICM_object = self.dataReader_object.get_ICM_from_name(ICM_name)
            ICM_object = ICM_object[ICM_filter_mask,:]

            ICM_mapper_name = "tokenToFeatureMapper_{}".format(ICM_name)
            ICM_mapper_object = self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name)

            ICM_object, _, ICM_mapper_object = removeFeatures(ICM_object, minOccurrence = 1, maxPercOccurrence = 1.00,
                                                                reconcile_mapper = ICM_mapper_object)

            setattr(self, ICM_name, ICM_object)
            setattr(self, ICM_mapper_name, ICM_mapper_object)



        print("DataReaderPostprocessing_User_min_interactions: Removing items from ICMs... done")







def select_users_with_min_interactions(URM, min_interactions = 5, reshape = False):
    """

    :param URM:
    :param min_interactions:
    :param reshape:
    :return: URM, removedUsers, removedItems
    """

    print("DataReaderPostprocessing_User_min_interactions: min_interactions extraction will zero out some users and items without changing URM shape")

    URM.eliminate_zeros()

    n_users = URM.shape[0]
    n_items = URM.shape[1]


    print("DataReaderPostprocessing_User_min_interactions: Initial URM desity is {:.2E}".format(URM.nnz/(n_users*n_items)))

    n_users, n_items = URM.shape

    URM = sps.csr_matrix(URM)
    user_to_remove_mask = np.ediff1d(URM.indptr) < min_interactions
    removed_users = np.arange(0, n_users, dtype=np.int)[user_to_remove_mask]


    for user in removed_users:
        start_pos = URM.indptr[user]
        end_pos = URM.indptr[user + 1]

        URM.data[start_pos:end_pos] = np.zeros_like(URM.data[start_pos:end_pos])

    URM.eliminate_zeros()

    URM = sps.csc_matrix(URM)
    items_to_remove_mask = np.ediff1d(URM.indptr) == 0
    removed_items = np.arange(0, n_items, dtype=np.int)[items_to_remove_mask]


    if URM.data.sum() == 0:
        print("DataReaderPostprocessing_User_min_interactions: WARNING URM is empty.")

    else:
         print("DataReaderPostprocessing_User_min_interactions: URM desity without zeroed-out nodes is {:.2E}.\n"
              "Users with less than {} interactions are {} ( {:.2f}%), Items are {} ( {:.2f}%)".format(
            sum(URM.data)/((n_users-len(removed_users))*(n_items-len(removed_items))),
            min_interactions,
            len(removed_users), len(removed_users)/n_users*100,
            len(removed_items), len(removed_items)/n_items*100))


    print("DataReaderPostprocessing_User_min_interactions: split complete")

    URM = sps.csr_matrix(URM)

    if reshape:
        # Remove all columns and rows with no interactions
        return remove_empty_rows_and_cols(URM)


    return URM.copy(), removed_users, removed_items
