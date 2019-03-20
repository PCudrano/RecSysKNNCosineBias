#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.DataReaderPostprocessing import DataReaderPostprocessing
import numpy as np



class DataReaderPostprocessing_Implicit_URM(DataReaderPostprocessing):
    """
    This class transforms the URM from explicit (or whatever data content it had) to implicit
    """


    def __init__(self, dataReader_object):
        super(DataReaderPostprocessing_Implicit_URM, self).__init__(dataReader_object)



    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "implicit/"

        inner_subfolder_name = self.dataReader_object._get_dataset_name_data_subfolder()

        # Avoid concatenating the original/ part
        if inner_subfolder_name != self.DATASET_SUBFOLDER_ORIGINAL:
            subfolder_name += inner_subfolder_name

        return subfolder_name



    def _replace_interactions_with_ones(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos= 0

        blockSize = 1000


        while end_pos < len(self.URM_all.data):

            end_pos = min(len(self.URM_all.data), end_pos + blockSize)

            self.URM_all.data[start_pos:end_pos] = np.ones(end_pos-start_pos)

            start_pos += blockSize




    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the k-cores
        :return:
        """

        self.dataReader_object.load_data()

        self.URM_all = self.dataReader_object.get_URM_all()
        self.item_original_ID_to_index = self.dataReader_object.item_original_ID_to_index
        self.user_original_ID_to_index = self.dataReader_object.user_original_ID_to_index

        self._replace_interactions_with_ones()


        print("DataReaderPostprocessing_Implicit_URM: Copying ICMs... ")

        loaded_ICMs = self.dataReader_object.get_loaded_ICM_names()

        for ICM_name in loaded_ICMs:

            print("DataReaderPostprocessing_Implicit_URM: Copying ICM {}".format(ICM_name))

            ICM_object = self.dataReader_object.get_ICM_from_name(ICM_name)

            ICM_mapper_name = "tokenToFeatureMapper_{}".format(ICM_name)
            ICM_mapper_object = self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name)

            setattr(self, ICM_name, ICM_object)
            setattr(self, ICM_mapper_name, ICM_mapper_object)


        print("DataReaderPostprocessing_Implicit_URM: Copying ICMs... done")



