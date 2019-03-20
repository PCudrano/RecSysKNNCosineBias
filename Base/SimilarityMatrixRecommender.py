#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

import pickle
import numpy as np



class SimilarityMatrixRecommender(object):
    """
    This class refers to a Recommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self):
        super(SimilarityMatrixRecommender, self).__init__()

        self._compute_item_score = self._compute_score_item_based



    def _compute_score_item_based(self, user_id_array, items_to_compute = None):

        self._check_sparse_format(self.URM_train, "csr", "URM_train")
        self._check_sparse_format(self.W_sparse, "csc", "W_sparse")

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = user_profile_array.dot(self.W_sparse[:,items_to_compute]).toarray()
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()

        return item_scores






    def _compute_score_user_based(self, user_id_array, items_to_compute = None):

        # URM_train must be CSR, so compute all predictions as it is often the faster
        self._check_sparse_format(self.W_sparse, "csr", "W_sparse")

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train).toarray()

        return item_scores








    def _check_sparse_format(self, sparse_matrix, format, matrix_label):

        attr_name = "_check_sparse_format_done_{}_{}_flag".format(sparse_matrix, format.upper())

        if hasattr(self, attr_name):
            return

        else:

            setattr(self, attr_name, True)

            if sparse_matrix.getformat() != format:
                print("compute_item_score: {} is not {}, this will significantly slow down the computation.".format(matrix_label, format.upper()))





    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"sparse_weights": self.sparse_weights}


        dictionary_to_save["W_sparse"] = self.W_sparse


        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))
