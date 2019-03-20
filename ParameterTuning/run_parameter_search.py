#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender




from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender


from skopt.space import Real, Integer, Categorical


import traceback
from Utils.PoolWithSubprocess import PoolWithSubprocess


from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters







def runParameterSearch_FeatureWeighting(recommender_class, URM_train, W_train, ICM_object, ICM_name, n_cases = 30,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                             evaluator_validation_earlystopping = None,
                             output_folder_path ="result_experiments/",
                             similarity_type_list = None):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)



   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    if recommender_class is FBSM_Rating_Cython:

        hyperparamethers_range_dictionary = {}
        hyperparamethers_range_dictionary["topK"] = Categorical([300])
        hyperparamethers_range_dictionary["n_factors"] = Integer(1, 5)

        hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
        hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adam"])
        hyperparamethers_range_dictionary["l2_reg_D"] = Real(low = 1e-6, high = 1e1, prior = 'log-uniform')
        hyperparamethers_range_dictionary["l2_reg_V"] = Real(low = 1e-6, high = 1e1, prior = 'log-uniform')
        hyperparamethers_range_dictionary["epochs"] = Categorical([300])


        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {"validation_every_n": 5,
                                "stop_on_validation": True,
                                "evaluator_object": evaluator_validation_earlystopping,
                                "lower_validations_allowed": 10,
                                "validation_metric": metric_to_optimize}
        )




    if recommender_class is CFW_D_Similarity_Cython:

        hyperparamethers_range_dictionary = {}
        hyperparamethers_range_dictionary["topK"] = Categorical([300])

        hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
        hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adam"])
        hyperparamethers_range_dictionary["l1_reg"] = Real(low = 1e-3, high = 1e-2, prior = 'log-uniform')
        hyperparamethers_range_dictionary["l2_reg"] = Real(low = 1e-3, high = 1e-1, prior = 'log-uniform')
        hyperparamethers_range_dictionary["epochs"] = Categorical([50])

        hyperparamethers_range_dictionary["init_type"] = Categorical(["one", "random"])
        hyperparamethers_range_dictionary["add_zeros_quota"] = Real(low = 0.50, high = 1.0, prior = 'uniform')
        hyperparamethers_range_dictionary["positive_only_weights"] = Categorical([True, False])
        hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True])

        hyperparamethers_range_dictionary["use_dropout"] = Categorical([True])
        hyperparamethers_range_dictionary["dropout_perc"] = Real(low = 0.30, high = 0.8, prior = 'uniform')


        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object, W_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {"precompute_common_features":False,     # Reduces memory requirements
                                "validation_every_n": 5,
                                "stop_on_validation": True,
                                "evaluator_object": evaluator_validation_earlystopping,
                                "lower_validations_allowed": 10,
                                "validation_metric": metric_to_optimize}
        )




    if recommender_class is CFW_DVV_Similarity_Cython:

        hyperparamethers_range_dictionary = {}
        hyperparamethers_range_dictionary["topK"] = Categorical([300])
        hyperparamethers_range_dictionary["n_factors"] = Integer(1, 2)

        hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-3, prior = 'log-uniform')
        hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adam"])
        hyperparamethers_range_dictionary["l2_reg_D"] = Real(low = 1e-6, high = 1e1, prior = 'log-uniform')
        hyperparamethers_range_dictionary["l2_reg_V"] = Real(low = 1e-6, high = 1e1, prior = 'log-uniform')
        hyperparamethers_range_dictionary["epochs"] = Categorical([100])

        hyperparamethers_range_dictionary["add_zeros_quota"] = Real(low = 0.50, high = 1.0, prior = 'uniform')


        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object, W_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {"precompute_common_features":False,     # Reduces memory requirements
                                "validation_every_n": 5,
                                "stop_on_validation": True,
                                "evaluator_object": evaluator_validation_earlystopping,
                                "lower_validations_allowed": 10,
                                "validation_metric": metric_to_optimize}
        )




    ## Final step, after the hyperparameter range has been defined for each type of algorithm
    parameterSearch.search(recommender_parameters,
                           parameter_search_space = hyperparamethers_range_dictionary,
                           n_cases = n_cases,
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root,
                           metric_to_optimize = metric_to_optimize)









def runParameterSearch_Hybrid(recommender_class, URM_train, ICM_object, ICM_name, n_cases = 30,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                             output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True,
                             similarity_type_list = None ):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)



   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    if recommender_class is ItemKNN_CFCBF_Hybrid_Recommender:

        if similarity_type_list is None:
            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]


        hyperparamethers_range_dictionary = {}
        hyperparamethers_range_dictionary["ICM_weight"] = Real(low = 1e-2, high = 1e2, prior = 'log-uniform')

        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [ICM_object, URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )


        run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                       parameter_search_space = hyperparamethers_range_dictionary,
                                                       recommender_parameters = recommender_parameters,
                                                       parameterSearch = parameterSearch,
                                                       n_cases = n_cases,
                                                       output_folder_path = output_folder_path,
                                                       output_file_name_root = output_file_name_root,
                                                       metric_to_optimize = metric_to_optimize,
                                                       allow_weighting = allow_weighting)



        if parallelizeKNN:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
            resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            pool.close()
            pool.join()

        else:

            for similarity_type in similarity_type_list:
                run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


        return












def run_KNNRecommender_on_similarity_type(similarity_type, parameterSearch,
                                          parameter_search_space,
                                          recommender_parameters,
                                          n_cases,
                                          output_folder_path,
                                          output_file_name_root,
                                          metric_to_optimize,
                                          allow_weighting = False,
                                          allow_bias_ICM = False):

    original_parameter_search_space = parameter_search_space

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
    hyperparamethers_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparamethers_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])

    is_set_similarity = similarity_type in ["tversky", "dice", "jaccard", "tanimoto"]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "euclidean":
        hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])
        hyperparamethers_range_dictionary["normalize_avg_row"] = Categorical([True, False])
        hyperparamethers_range_dictionary["similarity_from_distance_mode"] = Categorical(["lin", "log", "exp"])


    if not is_set_similarity:

        if allow_weighting:
            hyperparamethers_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

        if allow_bias_ICM:
            hyperparamethers_range_dictionary["ICM_bias"] = Real(low = 1e-2, high = 1e+3, prior = 'log-uniform')


    local_parameter_search_space = {**hyperparamethers_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_parameters,
                           parameter_search_space = local_parameter_search_space,
                           n_cases = n_cases,
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root + "_" + similarity_type,
                           metric_to_optimize = metric_to_optimize)





def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, n_cases = 30,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                             output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True,
                             similarity_type_list = None, allow_bias_ICM = False):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)





   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    if similarity_type_list is None:
        similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]


    recommender_parameters = SearchInputRecommenderParameters(
        CONSTRUCTOR_POSITIONAL_ARGS = [ICM_object, URM_train],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )


    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                   recommender_parameters = recommender_parameters,
                                                   parameter_search_space = {},
                                                   parameterSearch = parameterSearch,
                                                   n_cases = n_cases,
                                                   output_folder_path = output_folder_path,
                                                   output_file_name_root = output_file_name_root,
                                                   metric_to_optimize = metric_to_optimize,
                                                   allow_weighting = allow_weighting,
                                                   allow_bias_ICM = allow_bias_ICM)



    if parallelizeKNN:
        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

        pool.close()
        pool.join()

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)








def runParameterSearch_Collaborative(recommender_class, URM_train, metric_to_optimize = "PRECISION",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     output_folder_path ="result_experiments/", parallelizeKNN = True, n_cases = 35, allow_weighting = True,
                                     similarity_type_list = None):



    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)




        if recommender_class in [TopPop, GlobalEffects, Random]:
            """
            TopPop, GlobalEffects and Random have no parameters therefore only one evaluation is needed
            """


            parameterSearch = SearchSingleCase(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

            parameterSearch.search(recommender_parameters,
                                   fit_parameters_values={},
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = output_file_name_root
                                   )


            return



        ##########################################################################################################

        if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

            if similarity_type_list is None:
                similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                           recommender_parameters = recommender_parameters,
                                                           parameter_search_space = {},
                                                           parameterSearch = parameterSearch,
                                                           n_cases = n_cases,
                                                           output_folder_path = output_folder_path,
                                                           output_file_name_root = output_file_name_root,
                                                           metric_to_optimize = metric_to_optimize,
                                                           allow_weighting = allow_weighting)



            if parallelizeKNN:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
                pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

                pool.close()
                pool.join()

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


            return



       ##########################################################################################################

        if recommender_class is P3alphaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        ##########################################################################################################

        if recommender_class is RP3betaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["beta"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )



        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150)
            hyperparamethers_range_dictionary["reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {"validation_every_n": 5,
                                    "stop_on_validation": True,
                                    "evaluator_object": evaluator_validation_earlystopping,
                                    "lower_validations_allowed": 20,
                                    "validation_metric": metric_to_optimize}
            )



        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150)
            hyperparamethers_range_dictionary["batch_size"] = Categorical([1])
            hyperparamethers_range_dictionary["positive_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["negative_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {"validation_every_n": 5,
                                    "stop_on_validation": True,
                                    "evaluator_object": evaluator_validation_earlystopping,
                                    "lower_validations_allowed": 20,
                                    "validation_metric": metric_to_optimize,
                                    "positive_threshold_BPR": None}
            )


        ##########################################################################################################

        if recommender_class is PureSVDRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 250)

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            hyperparamethers_range_dictionary["lambda_i"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["lambda_j"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {'train_with_sparse_weights':None, 'symmetric':True},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {"validation_every_n": 5,
                                    "stop_on_validation": True,
                                    "evaluator_object": evaluator_validation_earlystopping,
                                    "lower_validations_allowed": 20,
                                    "validation_metric": metric_to_optimize,
                                    "positive_threshold_BPR": None}
            )



        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["l1_ratio"] = Real(low = 1e-5, high = 1.0, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )



       #########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_parameters,
                               parameter_search_space = hyperparamethers_range_dictionary,
                               n_cases = n_cases,
                               output_folder_path = output_folder_path,
                               output_file_name_root = output_file_name_root,
                               metric_to_optimize = metric_to_optimize)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()











import os, multiprocessing
from functools import partial






def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    from Data_manager.Movielens1M.Movielens1MReader import Movielens1MReader
    from Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold


    dataset_object = Movielens1MReader()

    dataSplitter = DataSplitter_Warm_k_fold(dataset_object)

    dataSplitter.load_data()

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()


    output_folder_path = "result_experiments/SKOPT_prova/"


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)







    collaborative_algorithm_list = [
        Random,
        TopPop,
        P3alphaRecommender,
        RP3betaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
    ]



    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])


    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       metric_to_optimize = "MAP",
                                                       n_cases = 8,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path)



    from Utils.PoolWithSubprocess import PoolWithSubprocess


    # pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
    # pool.close()
    # pool.join()



    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()








if __name__ == '__main__':


    read_data_split_and_search()
