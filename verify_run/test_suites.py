from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (DatasetsSizeComparison, DateTrainTestLeakageDuplicates,
                                       DateTrainTestLeakageOverlap, FeatureDrift, FeatureLabelCorrelationChange,
                                       IndexTrainTestLeakage, LabelDrift,
                                       MultivariateDrift,
                                       NewCategoryTrainTest, NewLabelTrainTest, StringMismatchComparison,
                                       TrainTestSamplesMix)
from deepchecks.tabular.suites import model_evaluation


def first_custom_suite():
    kwargs = {}
    return Suite(
            'My Train Test Validation Suite',
            DatasetsSizeComparison(**kwargs).add_condition_test_train_size_ratio_greater_than(),
            NewLabelTrainTest(**kwargs).add_condition_new_labels_number_less_or_equal(),
            NewCategoryTrainTest(**kwargs).add_condition_new_category_ratio_less_or_equal(),
            StringMismatchComparison(**kwargs).add_condition_no_new_variants(),
            DateTrainTestLeakageDuplicates(**kwargs).add_condition_leakage_ratio_less_or_equal(),
            DateTrainTestLeakageOverlap(**kwargs).add_condition_leakage_ratio_less_or_equal(),
            IndexTrainTestLeakage(**kwargs).add_condition_ratio_less_or_equal(),
            # TrainTestSamplesMix(**kwargs).add_condition_duplicates_ratio_less_or_equal(),
            FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_less_than()
            .add_condition_feature_pps_in_train_less_than(),
            FeatureDrift(**kwargs).add_condition_drift_score_less_than(),
            LabelDrift(**kwargs).add_condition_drift_score_less_than(),
            MultivariateDrift(**kwargs).add_condition_overall_drift_value_less_than(),
        )

def my_model_evaluation():
    return model_evaluation()