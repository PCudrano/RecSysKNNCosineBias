
from Data_manager.DataReader_ImportAll import *

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout



if __name__ == '__main__':


    from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

    dataset_object = Movielens1MReader()
    #
    dataSplitter = DataSplitter_leave_k_out(dataset_object, k_value=1, validation_set=True, leave_last_out=True)
    dataSplitter.load_data()

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    recommender = ItemKNNCFRecommender(URM_train)

    recommender.fit()


    evaluator = EvaluatorHoldout(URM_test, [5, 10, 15], exclude_seen=True)

    results_run, results_run_string = evaluator.evaluateRecommender(recommender)

    print("Original:\n" + results_run_string)
