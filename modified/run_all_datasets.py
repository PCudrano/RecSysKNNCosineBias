#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/03/19

@author: Maurizio Ferrari Dacrema
"""

import multiprocessing, traceback, os
from functools import  partial
# TODO ADDED
import skopt
import datetime
# TODO /ADDED

os.chdir('..')

from Data_manager.DataReader_ImportAll import *
from modified.ParameterTuning.run_parameter_search import runParameterSearch_Content, runParameterSearch_Collaborative
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


def run_on_dataset_ICM_name(ICM_name, dataset_class, similarity_type_list, output_folder, allow_bias_ICM,
                            # TODO ADDED
                            feature_weighting
                            # TODO /ADDED
                            ):

    try:

        print("Processing ICM: '{}'".format(ICM_name))

        dataset_object = dataset_class(ICM_to_load_list = [ICM_name])

        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_value=1, validation_set=True)
        dataSplitter.load_data()

        ICM_object = dataSplitter.get_ICM_from_name(ICM_name)

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 15, 20, 25, 30])


        runParameterSearch_Content(ItemKNNCBFRecommender,
                                   URM_train,
                                   ICM_object,
                                   ICM_name,
                                   n_cases = 40,
                                   evaluator_validation = evaluator_validation,
                                   evaluator_test = evaluator_test,
                                   metric_to_optimize = "MAP",
                                   output_folder_path = output_folder,
                                   parallelizeKNN = True,
                                   allow_weighting = feature_weighting,
                                   allow_bias_ICM = allow_bias_ICM,
                                   similarity_type_list = similarity_type_list)


    except Exception as e:

        print("On ICM {} Exception {}".format(ICM_name, str(e)))
        traceback.print_exc()





def run_on_dataset(dataset_class, allow_bias_ICM,
                   # TODO ADDED
                   feature_weighting
                   # TODO /ADDED
                   ):
    """
    :param dataset_class:
    :param allow_bias_ICM:
    :param feature_weighting: can either be bool True/False --> enable/disable optimizing wrt to all types of feature weighting
                              or string "none", "TF-IDF", "BM25" --> forces one type of feature weighting
    :return:
    """

    similarity_type_list = ["cosine"]


    output_folder = "result_experiments/{}".format(dataset_class.DATASET_SUBFOLDER)

    if allow_bias_ICM:
        output_folder += "ICM_bias/"
    else:
        output_folder += "ICM_original/"

    output_folder += "feature_weighting_"+feature_weighting+"/"

    # If directory does not exist, create
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    dataSplitter = DataSplitter_leave_k_out(dataset_class(), k_value=1, validation_set=True)
    dataSplitter.load_data()

    all_available_ICM_names = dataSplitter.get_loaded_ICM_names()


    run_on_dataset_ICM_name_partial = partial(run_on_dataset_ICM_name,
                                              dataset_class = dataset_class,
                                              similarity_type_list = similarity_type_list,
                                              output_folder = output_folder,
                                              allow_bias_ICM = allow_bias_ICM,
                                              # TODO ADDED
                                              feature_weighting=feature_weighting
                                              # TODO /ADDED
                                              )


    for ICM_name in all_available_ICM_names:
        run_on_dataset_ICM_name_partial(ICM_name)

    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()*2/3), maxtasksperchild=1)
    # pool.map(run_on_dataset_ICM_name_partial, all_available_ICM_names)
    #
    # pool.close()
    # pool.join()





























if __name__ == '__main__':

    dataset_class_list = [
        # Movielens100KReader, # NO ICM available
        # Movielens1MReader, # Done
        # Movielens10MReader, # Done
        # Movielens20MReader, # Done
        # TheMoviesDatasetReader, # Done
        #
        # BrightkiteReader, # NO ICM available
        # EpinionsReader, # NO ICM available
        # NetflixPrizeReader, # NO ICM available
        # ThirtyMusicReader, # NO ICM available
        # YelpReader, # NO ICM available
        # BookCrossingReader, # Done
        # NetflixEnhancedReader, # NO ICM available
        #
        # AmazonElectronicsReader, # Not done
        # AmazonBooksReader, # Not done
        # AmazonAutomotiveReader, # Not done
        # AmazonMoviesTVReader, # Not done
        #
        # SpotifyChallenge2018Reader, # NO ICM available
        # XingChallenge2016Reader, # Not done - Too many items
        # XingChallenge2017Reader, # Not done - Too many items
        #
        # TVAudienceReader, # NO ICM available
        # LastFMHetrec2011Reader, # Done
        # # DeliciousHetrec2011Reader,
        # CiteULike_aReader, # Done
        # CiteULike_tReader, # Done
        ]

    feature_weighting_list = [
        "none",
        "TF-IDF",
        "BM25"
    ]

    allow_bias_ICM_list = [
        False,
        True
    ]

    for dataset_class in dataset_class_list:
        for feature_weighting in feature_weighting_list:
            for allow_bias_ICM in allow_bias_ICM_list:
                run_on_dataset(dataset_class, allow_bias_ICM = allow_bias_ICM, feature_weighting=feature_weighting)
                # Obs: if one dataset has more ICM, all ICMs are used by run_on_dataset() before passing to next cycle heres

