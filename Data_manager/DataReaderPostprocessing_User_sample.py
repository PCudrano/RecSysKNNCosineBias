#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.DataReaderPostprocessing import DataReaderPostprocessing
from Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens, removeFeatures
from Data_manager.DataReaderPostprocessing_K_Cores import select_k_cores
import numpy as np



class DataReaderPostprocessing_User_sample(DataReaderPostprocessing):
    """
    This class selects a partition of URM such that only some of the original users are present
    """


    def __init__(self, dataReader_object, user_quota = 1.0):

        assert user_quota > 0.0 and user_quota <= 1.0,\
            "DataReaderPostprocessing_User_sample: user_quota must be a positive value > 0.0 and <= 1.0, provided value was {}".format(user_quota)

        super(DataReaderPostprocessing_User_sample, self).__init__(dataReader_object)

        self.user_quota = user_quota


    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "{}_user_sample/".format(self.user_quota)

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

        num_users_to_select = int(n_users*self.user_quota)

        print("DataReaderPostprocessing_User_sample: Sampling {:.2f} % of all users, their number is {}".format(self.user_quota*100, num_users_to_select))

        user_id_list = np.arange(0, n_users, dtype=np.int)

        np.random.shuffle(user_id_list)

        sampled_user_id_list = user_id_list[0:num_users_to_select]
        removedUsers = user_id_list[num_users_to_select:]


        sampled_user_flag = np.zeros_like(user_id_list, dtype=np.bool)
        sampled_user_flag[sampled_user_id_list] = True

        self.URM_all = self.URM_all[sampled_user_flag,:]

        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)


        # Apply K - core to remove items with no interactions
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = 1, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        print("DataReaderPostprocessing_User_sample: Removed {} users and {} items with no interactions".format(len(removedUsers), len(removedItems)))

        ICM_filter_mask = np.ones(n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False


        print("DataReaderPostprocessing_User_sample: Removing items from ICMs... ")


        loaded_ICMs = self.dataReader_object.get_loaded_ICM_names()

        for ICM_name in loaded_ICMs:

            print("DataReaderPostprocessing_User_sample: Removing items from {}".format(ICM_name))

            ICM_object = self.dataReader_object.get_ICM_from_name(ICM_name)
            ICM_object = ICM_object[ICM_filter_mask,:]

            ICM_mapper_name = "tokenToFeatureMapper_{}".format(ICM_name)
            ICM_mapper_object = self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name)

            ICM_object, _, ICM_mapper_object = removeFeatures(ICM_object, minOccurrence = 1, maxPercOccurrence = 1.00,
                                                                reconcile_mapper = ICM_mapper_object)

            setattr(self, ICM_name, ICM_object)
            setattr(self, ICM_mapper_name, ICM_mapper_object)



        print("DataReaderPostprocessing_User_sample: Removing items from ICMs... done")



