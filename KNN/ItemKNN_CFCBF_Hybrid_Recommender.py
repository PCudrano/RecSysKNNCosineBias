#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import scipy.sparse as sps


class ItemKNN_CFCBF_Hybrid_Recommender(ItemKNNCBFRecommender, SimilarityMatrixRecommender, Recommender):
    """ ItemKNN_CFCBF_Hybrid_Recommender"""

    RECOMMENDER_NAME = "ItemKNN_CFCBF_Hybrid_Recommender"

    def __init__(self, ICM, URM_train):
        super(ItemKNN_CFCBF_Hybrid_Recommender, self).__init__(ICM, URM_train)


    def fit(self, ICM_weight = 1.0, **fit_args):

        self.ICM = self.ICM*ICM_weight
        self.ICM = sps.hstack([self.ICM, self.URM_train.T], format='csr')

        super(ItemKNN_CFCBF_Hybrid_Recommender, self).fit(**fit_args)

