#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/03/19

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.DataReader_ImportAll import *
import multiprocessing, traceback, os
from functools import  partial

from ParameterTuning.run_parameter_search import runParameterSearch_Content, runParameterSearch_Collaborative
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


def run_on_dataset_ICM_name(ICM_name, dataset_class, similarity_type_list, output_folder, allow_bias_ICM):

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
                                   allow_weighting = False,
                                   allow_bias_ICM = allow_bias_ICM,
                                   similarity_type_list = similarity_type_list)




    except Exception as e:

        print("On ICM {} Exception {}".format(ICM_name, str(e)))
        traceback.print_exc()





def run_on_dataset(dataset_class, allow_bias_ICM):

    similarity_type_list = ["cosine"]


    output_folder = "result_experiments/{}".format(dataset_class.DATASET_SUBFOLDER)

    if allow_bias_ICM:
        output_folder += "ICM_bias/"
    else:
        output_folder += "ICM_original/"


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
                                              allow_bias_ICM = allow_bias_ICM)


    for ICM_name in all_available_ICM_names:
        run_on_dataset_ICM_name_partial(ICM_name)

    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()*2/3), maxtasksperchild=1)
    # pool.map(run_on_dataset_ICM_name_partial, all_available_ICM_names)
    #
    # pool.close()
    # pool.join()





























if __name__ == '__main__':

    dataset_class_list = [
        # Movielens100KReader,
        # Movielens1MReader,
        # Movielens10MReader,
        # Movielens20MReader,
        # TheMoviesDatasetReader,
        #
        # BrightkiteReader,
        # EpinionsReader,
        # NetflixPrizeReader,
        # ThirtyMusicReader,
        # YelpReader,
        # BookCrossingReader,
        # NetflixEnhancedReader,
        #
        # AmazonElectronicsReader,
        # AmazonBooksReader,
        # AmazonAutomotiveReader,
        # AmazonMoviesTVReader,
        #
        # SpotifyChallenge2018Reader,
        # XingChallenge2016Reader,
        # XingChallenge2017Reader,
        #
        # TVAudienceReader,
        # LastFMHetrec2011Reader,
        # DeliciousHetrec2011Reader,
        CiteULike_aReader,
        CiteULike_tReader,
        ]



    for dataset_class in dataset_class_list:
        run_on_dataset(dataset_class, allow_bias_ICM = False)
        run_on_dataset(dataset_class, allow_bias_ICM = True)