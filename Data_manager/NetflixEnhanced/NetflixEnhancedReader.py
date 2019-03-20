#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import zipfile, os

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import removeFeatures, reconcile_mapper_with_removed_tokens
from Data_manager.DataReaderPostprocessing_K_Cores import select_k_cores


class NetflixEnhancedReader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EYHOBNp2nF1Gtvm2ELd9eLkBJfwKvU2O4Cp9HjO6HUJkhA?e=I2S1OC"
    DATASET_SUBFOLDER = "NetflixEnhanced/"
    AVAILABLE_ICM = ["ICM_all", "ICM_tags", "ICM_editorial"]


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("NetflixEnhancedReader: Loading original data")

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + "NetflixEnhancedData.zip")

            URM_matfile_path = dataFile.extract("urm.mat", path=decompressed_zip_file_folder + "decompressed/")
            titles_matfile_path = dataFile.extract("titles.mat", path=decompressed_zip_file_folder + "decompressed/")
            ICM_matfile_path = dataFile.extract("icm.mat", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("NetflixPrizeReader: Unable to find or extract data zip file.")
            print("NetflixPrizeReader: Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            print("NetflixPrizeReader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")





        URM_matfile = sio.loadmat(URM_matfile_path)

        self.URM_all = URM_matfile["urm"]
        usercache_urm = URM_matfile["usercache_urm"]
        itemcache_urm = URM_matfile["itemcache_urm"]


        self.item_original_ID_to_index = {}

        for item_id in range(self.URM_all.shape[1]):
            self.item_original_ID_to_index[item_id] = item_id

        for user_id in range(self.URM_all.shape[0]):
            self.user_original_ID_to_index[user_id] = user_id



        titles_matfile = sio.loadmat(titles_matfile_path)

        titles_list = titles_matfile["titles"]


        ICM_matfile = sio.loadmat(ICM_matfile_path)

        self.ICM_all = ICM_matfile["icm"]
        self.ICM_all = sps.csr_matrix(self.ICM_all.T)

        ICM_dictionary = ICM_matfile["dictionary"]
        itemcache_icm = ICM_matfile["itemcache_icm"]
        stemTypes = ICM_dictionary["stemTypes"][0][0]
        stems = ICM_dictionary["stems"][0][0]

        self.tokenToFeatureMapper_ICM_all = {}
        self.tokenToFeatureMapper_ICM_tags = {}
        self.tokenToFeatureMapper_ICM_editorial = {}

        # Split ICM_tags and ICM_editorial
        is_tag_mask = np.zeros((len(stems)), dtype=np.bool)

        for current_stem_index in range(len(stems)):
            current_stem_type = stemTypes[current_stem_index]
            current_stem_type_string = current_stem_type[0][0]

            token = stems[current_stem_index][0][0]

            if token in self.tokenToFeatureMapper_ICM_all:
                print("Duplicate token {} alredy existent in position {}".format(token, self.tokenToFeatureMapper_ICM_all[token]))

            else:
                self.tokenToFeatureMapper_ICM_all[token] = current_stem_index

                if "KeywordsArray" in current_stem_type_string:
                    is_tag_mask[current_stem_index] = True

                    self.tokenToFeatureMapper_ICM_tags[token] = len(self.tokenToFeatureMapper_ICM_tags)

                else:
                    self.tokenToFeatureMapper_ICM_editorial[token] = len(self.tokenToFeatureMapper_ICM_editorial)


        self.ICM_tags = self.ICM_all[:,is_tag_mask]


        is_editorial_mask = np.logical_not(is_tag_mask)
        self.ICM_editorial = self.ICM_all[:, is_editorial_mask]


        # Eliminate items and users with no interactions or less than the desired value
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = 1, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        n_items = self.ICM_all.shape[0]
        ICM_filter_mask = np.ones(n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False

        # Remove items in ICM as well to ensure consistency
        self.ICM_all = self.ICM_all[ICM_filter_mask,:]
        self.ICM_tags = self.ICM_tags[ICM_filter_mask,:]
        self.ICM_editorial = self.ICM_editorial[ICM_filter_mask,:]


        # Remove features taking into account the filtered ICM
        self.ICM_all, _, self.tokenToFeatureMapper_ICM_all = removeFeatures(self.ICM_all, minOccurrence = 5,
                                                                            maxPercOccurrence = 0.30, reconcile_mapper = self.tokenToFeatureMapper_ICM_all)
        self.ICM_tags, _, self.tokenToFeatureMapper_ICM_tags = removeFeatures(self.ICM_tags, minOccurrence = 5,
                                                                            maxPercOccurrence = 0.30, reconcile_mapper=self.tokenToFeatureMapper_ICM_tags)
        self.ICM_editorial, _, self.tokenToFeatureMapper_ICM_editorial = removeFeatures(self.ICM_editorial, minOccurrence = 5,
                                                                            maxPercOccurrence = 0.30, reconcile_mapper=self.tokenToFeatureMapper_ICM_editorial)




        print("NetflixEnhancedReader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_zip_file_folder + "decompressed", ignore_errors=True)

        print("NetflixEnhancedReader: loading complete")