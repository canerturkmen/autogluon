from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from autogluon.common.utils.log_utils import convert_time_in_s_to_log_friendly
from autogluon.core.constants import AUTO_WEIGHT, BALANCE_WEIGHT, BINARY, MULTICLASS, QUANTILE, REGRESSION
from autogluon.core.data import LabelCleaner
from autogluon.core.data.cleaner import Cleaner
from autogluon.core.utils.time import sample_df_for_time_func, time_func
from autogluon.core.utils.utils import augment_rare_classes, extract_column

from ..trainer import AutoTrainer
from .abstract_learner import AbstractTabularLearner

logger = logging.getLogger(__name__)


# TODO: Add functionality for advanced feature generators such as gl_code_matrix_generator (inter-row dependencies, apply to train differently than test, etc., can only run after train/test split, rerun for each cv fold)
# TODO: - Differentiate between advanced generators that require fit (stateful, gl_code_matrix) and those that do not (bucket label averaging in SCOT GC 2019)
# TODO: - Those that do not could be added to preprocessing function of model, but would then have to be recomputed on each model.
# TODO: Add cv / OOF generator option, so that AutoGluon can be used as a base model in an ensemble stacker
# Learner encompasses full problem, loading initial data, feature generation, model training, model prediction
class DefaultLearner(AbstractTabularLearner):
    def __init__(self, trainer_type=AutoTrainer, **kwargs):
        super().__init__(**kwargs)
        self.trainer_type = trainer_type
        self.class_weights = None
        self._time_fit_total = None
        self._time_fit_preprocessing = None
        self._time_fit_training = None
        self._time_limit = None
        self.preprocess_1_time = None  # Time required to preprocess 1 row of data
        self.preprocess_1_batch_size = None  # Batch size used to calculate self.preprocess_1_time

    # TODO: v0.1 Document trainer_fit_kwargs
    def _fit(
        self,
        X: DataFrame,
        X_val: DataFrame | None = None,
        X_test: DataFrame | None = None,
        X_unlabeled: DataFrame | None = None,
        holdout_frac: float = 0.1,
        num_bag_folds: int = 0,
        num_bag_sets: int = 1,
        time_limit: float | None = None,
        infer_limit: float | None = None,
        infer_limit_batch_size: int | None = None,
        verbosity: int = 2,
        raise_on_model_failure: bool = False,
        **trainer_fit_kwargs,
    ):
        """Arguments:
        X (DataFrame): training data
        X_val (DataFrame): data used for hyperparameter tuning. Note: final model may be trained using this data as well as training data
        X_test (DataFrame): data used for tracking model performance on test data during training. Note: this data is never used to train the model
        X_unlabeled (DataFrame): data used for pretraining a model. This is same data format as X, without label-column. This data is used for semi-supervised learning.
        holdout_frac (float): Fraction of data to hold out for evaluating validation performance (ignored if X_val != None, ignored if kfolds != 0)
        num_bag_folds (int): kfolds used for bagging of models, roughly increases model training time by a factor of k (0: disabled)
        num_bag_sets (int): number of repeats of kfold bagging to perform (values must be >= 1),
            total number of models trained during bagging = num_bag_folds * num_bag_sets
        """
        # TODO: if provided, feature_types in X, X_val are ignored right now, need to pass to Learner/trainer and update this documentation.
        self._time_limit = time_limit
        if time_limit:
            logger.log(20, f"Beginning AutoGluon training ... Time limit = {time_limit:.0f}s")
        else:
            logger.log(20, "Beginning AutoGluon training ...")
        logger.log(20, f'AutoGluon will save models to "{self.path}"')
        logger.log(20, f"Train Data Rows:    {len(X)}")
        logger.log(20, f"Train Data Columns: {len([column for column in X.columns if column != self.label])}")
        if X_val is not None:
            logger.log(20, f"Tuning Data Rows:    {len(X_val)}")
            logger.log(20, f"Tuning Data Columns: {len([column for column in X_val.columns if column != self.label])}")
        logger.log(20, f"Label Column:       {self.label}")
        time_preprocessing_start = time.time()
        self._pre_X_rows = len(X)
        if self.problem_type is None:
            self.problem_type = self.infer_problem_type(y=X[self.label])
        logger.log(20, f"Problem Type:       {self.problem_type}")
        if self._eval_metric_was_str:
            # Ensure that the eval_metric is valid for the problem_type
            self._verify_metric(eval_metric=self.eval_metric, problem_type=self.problem_type)
        if self.groups is not None:
            num_bag_sets = 1
            num_bag_folds = len(X[self.groups].unique())
        X_og = None if infer_limit_batch_size is None else X
        logger.log(20, "Preprocessing data ...")
        X, y, X_val, y_val, X_test, y_test, X_unlabeled, holdout_frac, num_bag_folds, groups = self.general_data_processing(
            X=X, X_val=X_val, X_test=X_test, X_unlabeled=X_unlabeled, holdout_frac=holdout_frac, num_bag_folds=num_bag_folds
        )
        if X_og is not None:
            infer_limit = self._update_infer_limit(X=X_og, infer_limit_batch_size=infer_limit_batch_size, infer_limit=infer_limit)

        self._post_X_rows = len(X)
        time_preprocessing_end = time.time()
        self._time_fit_preprocessing = time_preprocessing_end - time_preprocessing_start
        logger.log(20, f"Data preprocessing and feature engineering runtime = {round(self._time_fit_preprocessing, 2)}s ...")
        if time_limit:
            time_limit_trainer = time_limit - self._time_fit_preprocessing
        else:
            time_limit_trainer = None

        trainer = self.trainer_type(
            path=self.model_context,
            problem_type=self.label_cleaner.problem_type_transform,
            eval_metric=self.eval_metric,
            num_classes=self.label_cleaner.num_classes,
            quantile_levels=self.quantile_levels,
            feature_metadata=self.feature_generator.feature_metadata,
            low_memory=True,
            k_fold=num_bag_folds,  # TODO: Consider moving to fit call
            n_repeats=num_bag_sets,  # TODO: Consider moving to fit call
            sample_weight=self.sample_weight,
            weight_evaluation=self.weight_evaluation,
            save_data=self.cache_data,
            random_state=self.random_state,
            verbosity=verbosity,
            raise_on_model_failure=raise_on_model_failure,
        )

        self.trainer_path = trainer.path
        if self.eval_metric is None:
            self.eval_metric = trainer.eval_metric

        self.save()
        trainer.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_unlabeled=X_unlabeled,
            holdout_frac=holdout_frac,
            time_limit=time_limit_trainer,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            groups=groups,
            **trainer_fit_kwargs,
        )
        self.save_trainer(trainer=trainer)
        time_end = time.time()
        self._time_fit_training = time_end - time_preprocessing_end
        self._time_fit_total = time_end - time_preprocessing_start
        log_throughput = ""
        if trainer.model_best is not None:
            predict_n_time_per_row = trainer.get_model_attribute_full(model=trainer.model_best, attribute="predict_n_time_per_row")
            predict_n_size = trainer.get_model_attribute_full(model=trainer.model_best, attribute="predict_n_size", func=min)
            if predict_n_time_per_row is not None and predict_n_size is not None:
                log_throughput = f" | Estimated inference throughput: {1/(predict_n_time_per_row if predict_n_time_per_row else np.finfo(np.float16).eps):.1f} rows/s ({int(predict_n_size)} batch size)"
        logger.log(
            20, f"AutoGluon training complete, total runtime = {round(self._time_fit_total, 2)}s ... Best model: {trainer.model_best}" f"{log_throughput}"
        )

    def _update_infer_limit(self, X: DataFrame, *, infer_limit_batch_size: int, infer_limit: float = None):
        """
        Calculates preprocessing time per row for a given unprocessed data X and infer_limit_batch_size
        Returns an updated infer_limit if not None with preprocessing time per row subtracted
        Raises an exception if preprocessing time is greater than or equal to the infer_limit
        """
        X_batch = sample_df_for_time_func(df=X, sample_size=infer_limit_batch_size)
        infer_limit_batch_size_actual = len(X_batch)
        self.preprocess_1_time = time_func(f=self.transform_features, args=[X_batch]) / infer_limit_batch_size_actual
        self.preprocess_1_batch_size = infer_limit_batch_size
        preprocess_1_time_log, time_unit_preprocess_1_time = convert_time_in_s_to_log_friendly(self.preprocess_1_time)
        logger.log(
            20, f"\t{round(preprocess_1_time_log, 3)}{time_unit_preprocess_1_time}\t= Feature Preprocessing Time (1 row | {infer_limit_batch_size} batch size)"
        )

        if infer_limit is not None:
            infer_limit_new = infer_limit - self.preprocess_1_time
            infer_limit_log, time_unit_infer_limit = convert_time_in_s_to_log_friendly(infer_limit)
            infer_limit_new_log, time_unit_infer_limit_new = convert_time_in_s_to_log_friendly(infer_limit_new)

            logger.log(
                20,
                f"\t\tFeature Preprocessing requires {round(self.preprocess_1_time/infer_limit*100, 2)}% "
                f"of the overall inference constraint ({infer_limit_log}{time_unit_infer_limit})\n"
                f"\t\t{round(infer_limit_new_log, 3)}{time_unit_infer_limit_new} inference time budget remaining for models...",
            )
            if infer_limit_new <= 0:
                infer_limit_new = 0
                logger.log(
                    30,
                    f"WARNING: Impossible to satisfy inference constraint, budget is exceeded during data preprocessing!\n"
                    f"\tAutoGluon will be unable to satisfy the constraint, but will return the fastest model it can.\n"
                    f"\tConsider using fewer features, relaxing the inference constraint, or simplifying the feature generator.",
                )
            infer_limit = infer_limit_new
        return infer_limit

    # TODO: Add default values to X_val, X_unlabeled, holdout_frac, and num_bag_folds
    def general_data_processing(
        self, X: DataFrame, X_val: DataFrame = None, X_test: DataFrame = None, X_unlabeled: DataFrame = None, holdout_frac: float = 1, num_bag_folds: int = 0
    ):
        """General data processing steps used for all models."""
        X = self._check_for_non_finite_values(X, name="train", is_train=True)
        if X_val is not None:
            X_val = self._check_for_non_finite_values(X_val, name="val", is_train=False)
        if X_test is not None:
            X_test = self._check_for_non_finite_values(X_test, name="test", is_train=False)

        holdout_frac_og = holdout_frac
        if X_val is not None and self.label in X_val.columns:
            holdout_frac = 1

        if self.eval_metric is not None and self.eval_metric.needs_proba and self.problem_type == MULTICLASS:
            # Metric requires all classes present in training to be able to compute a score
            if num_bag_folds > 0:
                self.threshold = 2
                if self.groups is None:
                    X = augment_rare_classes(X, self.label, threshold=2)
            else:
                self.threshold = 1

        self.threshold, holdout_frac, num_bag_folds = self.adjust_threshold_if_necessary(
            X[self.label], threshold=self.threshold, holdout_frac=holdout_frac, num_bag_folds=num_bag_folds
        )

        # Gets labels prior to removal of infrequent classes
        y_uncleaned = X[self.label].copy()

        self.cleaner = Cleaner.construct(problem_type=self.problem_type, label=self.label, threshold=self.threshold)
        X = self.cleaner.fit_transform(X)  # TODO: Consider merging cleaner into label_cleaner
        X, y = self.extract_label(X)
        self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y, y_uncleaned=y_uncleaned, positive_class=self._positive_class)
        y = self.label_cleaner.transform(y)
        X = self.set_predefined_weights(X, y)
        X, w = extract_column(X, self.sample_weight)
        X, groups = extract_column(X, self.groups)
        if self.label_cleaner.num_classes is not None and self.problem_type != BINARY:
            logger.log(20, f"Train Data Class Count: {self.label_cleaner.num_classes}")

        X_val, y_val, w_val, holdout_frac = self._apply_cleaner_transform(
            X=X_val, y_uncleaned=y_uncleaned, holdout_frac=holdout_frac, holdout_frac_og=holdout_frac_og, name="val", is_test=False
        )
        X_test, y_test, w_test, _ = self._apply_cleaner_transform(
            X=X_test, y_uncleaned=y_uncleaned, holdout_frac=holdout_frac, holdout_frac_og=holdout_frac_og, name="test", is_test=True
        )

        self._original_features = list(X.columns)
        # TODO: Move this up to top of data before removing data, this way our feature generator is better
        logger.log(20, f"Using Feature Generators to preprocess the data ...")

        if X_test is not None:
            logger.log(
                15,
                "Performing general data preprocessing with merged train & validation data, so validation/test performance may not accurately reflect performance on new test data",
            )

        # TODO: extend this boolean flag into learner init parameter
        transform_with_test = False

        X_test_super = None
        y_test_super = None
        if transform_with_test:
            X_test_super = X_test
            y_test_super = y_test

        datasets = [X, X_val, X_test_super, X_unlabeled]
        X_super = pd.concat(datasets, ignore_index=True)

        if self.feature_generator.is_fit():
            logger.log(
                20,
                f"{self.feature_generator.__class__.__name__} is already fit, so the training data will be processed via .transform() instead of .fit_transform().",
            )
            X_super = self.feature_generator.transform(X_super)
            if not transform_with_test and X_test is not None:
                X_test = self.feature_generator.transform(X_test)
            self.feature_generator.print_feature_metadata_info()
        else:
            y_unlabeled = pd.Series(np.nan, index=X_unlabeled.index) if X_unlabeled is not None else None
            y_list = [y, y_val, y_test_super, y_unlabeled]
            y_super = pd.concat(y_list, ignore_index=True)
            X_super = self.fit_transform_features(X_super, y_super, problem_type=self.label_cleaner.problem_type_transform, eval_metric=self.eval_metric)
            if not transform_with_test and X_test is not None:
                X_test = self.feature_generator.transform(X_test)

        idx = 0
        for i in range(len(datasets)):
            if datasets[i] is not None:
                length = len(datasets[i])
                datasets[i] = X_super.iloc[idx : idx + length].set_index(datasets[i].index)
                idx += length

        X, X_val, X_test_super, X_unlabeled = datasets
        del X_super

        if transform_with_test:
            X_test = X_test_super

        # TODO: consider not bundling sample-weights inside X, X_val
        X = self.bundle_weights(X, w, "X", is_train=True)
        X_val = self.bundle_weights(X_val, w_val, "X_val", is_train=False)
        X_test = self.bundle_weights(X_test, w_test, "X_test", is_train=False)
        return X, y, X_val, y_val, X_test, y_test, X_unlabeled, holdout_frac, num_bag_folds, groups

    def bundle_weights(self, X: DataFrame | None, w: Series | None, name: str, is_train=False) -> DataFrame:
        if is_train:
            if w is not None:
                X[self.sample_weight] = w
        elif X is not None:
            if w is not None:
                X[self.sample_weight] = w
            elif not self.weight_evaluation:
                nan_vals = np.empty((len(X),))
                nan_vals[:] = np.nan
                X[self.sample_weight] = nan_vals
            else:
                raise ValueError(
                    f"sample_weight column '{self.sample_weight}' \
                                 cannot be missing from {name} dataset if weight_evaluation=True"
                )

        return X

    def set_predefined_weights(self, X, y):
        if self.sample_weight not in [AUTO_WEIGHT, BALANCE_WEIGHT] or self.problem_type not in [BINARY, MULTICLASS]:
            return X
        if self.sample_weight in X.columns:
            raise ValueError(
                f"Column name '{self.sample_weight}' cannot appear in your dataset with predefined weighting strategy. Please change it and try again."
            )
        if self.sample_weight == BALANCE_WEIGHT:
            if self.class_weights is None:
                class_counts = y.value_counts()
                n = len(y)
                k = len(class_counts)
                self.class_weights = {c: n / (class_counts[c] * k) for c in class_counts.index}
                logger.log(20, "Assigning sample weights to balance differences in frequency of classes.")
                logger.log(15, f"Balancing classes via the following weights: {self.class_weights}")
            w = y.map(self.class_weights)
        elif self.sample_weight == AUTO_WEIGHT:  # TODO: support more sophisticated auto_weight strategy
            raise NotImplementedError(f"{AUTO_WEIGHT} strategy not yet supported.")
        X[self.sample_weight] = w  # TODO: consider not bundling sample weights inside X
        return X

    def _check_for_non_finite_values(self, X: DataFrame, name: str = "", is_train: bool = False) -> DataFrame:
        if is_train or (X is not None and self.label in X.columns):
            X = copy.deepcopy(X)

            # treat None, NaN, INF, NINF as NA
            X[self.label] = X[self.label].replace([np.inf, -np.inf], np.nan)
            invalid_labels = X[self.label].isna()
            if invalid_labels.any():
                first_invalid_label_idx = invalid_labels.idxmax()
                raise ValueError(
                    f"{name} dataset label column cannot contain non-finite values (NaN, Inf, Ninf). First invalid label at data idx: {first_invalid_label_idx}"
                )

        return X

    def _apply_cleaner_transform(
        self, X: DataFrame, y_uncleaned: Series, holdout_frac: float | int, holdout_frac_og: float | int, name: str, is_test: bool = False
    ) -> tuple[DataFrame, Series, Series | None, float | int]:
        if X is not None and self.label in X.columns:
            y_og = X[self.label]
            len_og = len(X) if is_test else None
            X = self.cleaner.transform(X)
            if is_test and len(X) != len_og:
                # FIXME: Currently, there are ways in which this code can be reached (@innixma)
                raise AssertionError(
                    f"{name} cannot have low frequency classes! Please create a GitHub issue if you see this message, as it should never occur."
                )

            if len(X) == 0:
                logger.warning(
                    "############################################################################################################\n"
                    f"WARNING: All {name} data contained low frequency classes, ignoring {name} and generating from subset of X\n"
                    "\tYour input validation data or training data labels might be corrupted, please manually inspect them for correctness!"
                )
                if self.problem_type in [BINARY, MULTICLASS]:
                    train_classes = sorted(list(y_uncleaned.unique()))
                    val_classes = sorted(list(y_og.unique()))
                    logger.warning(f"\ttrain Classes: {train_classes}")
                    logger.warning(f"\t{name}   Classes: {val_classes}")
                    logger.warning(f"\ttrain Class Dtype: {y_uncleaned.dtype}")
                    logger.warning(f"\t{name}   Class Dtype: {y_og.dtype}")
                    missing_classes = [c for c in val_classes if c not in train_classes]
                    logger.warning(f"\tClasses missing from Training Data: {missing_classes}")
                logger.warning("############################################################################################################")

                X = None
                y = None
                w = None
                holdout_frac = holdout_frac_og
            else:
                X, y = self.extract_label(X)
                y = self.label_cleaner.transform(y)
                X = self.set_predefined_weights(X, y)
                X, w = extract_column(X, self.sample_weight)
        else:
            y = None
            w = None

        return X, y, w, holdout_frac

    def adjust_threshold_if_necessary(self, y, threshold, holdout_frac, num_bag_folds):
        new_threshold, new_holdout_frac, new_num_bag_folds = self._adjust_threshold_if_necessary(y, threshold, holdout_frac, num_bag_folds)
        if new_threshold != threshold:
            if new_threshold < threshold:
                logger.warning(f"Warning: Updated label_count_threshold from {threshold} to {new_threshold} to avoid cutting too many classes.")
        if new_holdout_frac != holdout_frac:
            if new_holdout_frac > holdout_frac:
                logger.warning(f"Warning: Updated holdout_frac from {holdout_frac} to {new_holdout_frac} to avoid cutting too many classes.")
        if new_num_bag_folds != num_bag_folds:
            logger.warning(f"Warning: Updated num_bag_folds from {num_bag_folds} to {new_num_bag_folds} to avoid cutting too many classes.")
        return new_threshold, new_holdout_frac, new_num_bag_folds

    def _adjust_threshold_if_necessary(self, y, threshold, holdout_frac, num_bag_folds):
        new_threshold = threshold
        num_rows = len(y)
        holdout_frac = max(holdout_frac, 1 / num_rows + 0.001)
        num_bag_folds = min(num_bag_folds, num_rows)

        if num_bag_folds < 2:
            minimum_safe_threshold = 1
        else:
            minimum_safe_threshold = 2

        if minimum_safe_threshold > new_threshold:
            new_threshold = minimum_safe_threshold

        if self.problem_type in [REGRESSION, QUANTILE]:
            return new_threshold, holdout_frac, num_bag_folds

        class_counts = y.value_counts()
        total_rows = class_counts.sum()
        minimum_percent_to_keep = 0.975
        minimum_rows_to_keep = math.ceil(total_rows * minimum_percent_to_keep)
        minimum_class_to_keep = 2

        num_classes = len(class_counts)
        class_counts_valid = class_counts[class_counts >= new_threshold]
        num_rows_valid = class_counts_valid.sum()
        num_classes_valid = len(class_counts_valid)

        if (num_rows_valid >= minimum_rows_to_keep) and (num_classes_valid >= minimum_class_to_keep):
            return new_threshold, holdout_frac, num_bag_folds

        num_classes_valid = 0
        num_rows_valid = 0
        new_threshold = None
        for i in range(num_classes):
            num_classes_valid += 1
            num_rows_valid += class_counts.iloc[i]
            new_threshold = class_counts.iloc[i]
            if (num_rows_valid >= minimum_rows_to_keep) and (num_classes_valid >= minimum_class_to_keep):
                break

        return new_threshold, holdout_frac, num_bag_folds

    def get_info(self, include_model_info=False, include_model_failures=False, **kwargs):
        learner_info = super().get_info(**kwargs)
        trainer = self.load_trainer()
        trainer_info = trainer.get_info(include_model_info=include_model_info, include_model_failures=include_model_failures)
        learner_info.update(
            {
                "time_fit_preprocessing": self._time_fit_preprocessing,
                "time_fit_training": self._time_fit_training,
                "time_fit_total": self._time_fit_total,
                "time_limit": self._time_limit,
            }
        )

        learner_info.update(trainer_info)
        return learner_info
