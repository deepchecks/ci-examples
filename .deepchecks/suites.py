from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (DatasetsSizeComparison, DateTrainTestLeakageDuplicates,
                                       DateTrainTestLeakageOverlap, FeatureDrift, FeatureLabelCorrelationChange,
                                       IndexTrainTestLeakage, LabelDrift,
                                       MultivariateDrift,
                                       NewCategoryTrainTest, NewLabelTrainTest, StringMismatchComparison,
                                       TrainTestSamplesMix)

# import all:

from deepchecks.tabular.checks import (BoostingOverfit, CalibrationScore, ConflictingLabels, ConfusionMatrixReport,
                                       DataDuplicates, DatasetsSizeComparison, DateTrainTestLeakageDuplicates,
                                       DateTrainTestLeakageOverlap, FeatureDrift, FeatureFeatureCorrelation,
                                       FeatureLabelCorrelation, FeatureLabelCorrelationChange,
                                       IdentifierLabelCorrelation, IndexTrainTestLeakage, IsSingleValue, LabelDrift,
                                       MixedDataTypes, MixedNulls, ModelInferenceTime, MultivariateDrift,
                                       NewCategoryTrainTest, NewLabelTrainTest, OutlierSampleDetection, PercentOfNulls,
                                       PredictionDrift, RegressionErrorDistribution, RocReport, SimpleModelComparison,
                                       SingleDatasetPerformance, SpecialCharacters, StringLengthOutOfBounds,
                                       StringMismatch, StringMismatchComparison, TrainTestPerformance,
                                       TrainTestSamplesMix, UnusedFeatures, WeakSegmentsPerformance)
from deepchecks.tabular.suites import model_evaluation


def my_full_model_evaluation():
    return model_evaluation()


def my_model_evaluation():
    kwargs = {}
    return Suite(
        'My Model Evaluation Suite',
        TrainTestPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than(),
        RocReport(**kwargs).add_condition_auc_greater_than(),
        ConfusionMatrixReport(**kwargs),
        PredictionDrift(**kwargs).add_condition_drift_score_less_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_greater_than(),
        # WeakSegmentsPerformance(**kwargs).add_condition_segments_relative_performance_greater_than(),
        CalibrationScore(**kwargs),
        RegressionErrorDistribution(
            **kwargs).add_condition_kurtosis_greater_than().add_condition_systematic_error_ratio_to_rmse_less_than(),
        UnusedFeatures(**kwargs).add_condition_number_of_high_variance_unused_features_less_or_equal(),
        BoostingOverfit(**kwargs).add_condition_test_score_percent_decline_less_than(),
        ModelInferenceTime(**kwargs).add_condition_inference_time_less_than(),
    )






def validation_custom_suite():
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