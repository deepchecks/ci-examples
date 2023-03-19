from deepchecks.tabular import Suite
from deepchecks.tabular.checks import *


def my_model_evaluation():
    kwargs = {}
    return Suite(
        'My Model Evaluation Suite',
        TrainTestPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than(),
        RocReport(**kwargs).add_condition_auc_greater_than(),
        ConfusionMatrixReport(**kwargs),
        PredictionDrift(**kwargs).add_condition_drift_score_less_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_greater_than(),
        CalibrationScore(**kwargs),
        UnusedFeatures(**kwargs).add_condition_number_of_high_variance_unused_features_less_or_equal(),
        BoostingOverfit(**kwargs).add_condition_test_score_percent_decline_less_than(),
        ModelInferenceTime(**kwargs).add_condition_inference_time_less_than(),
    )
