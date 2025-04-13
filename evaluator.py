from typing import Literal, List, Callable, Dict
import pandas as pd
from data import Dataset
from metrics import log_transformed_correlation_score, k_means_score, ml_efficiency_cat, regularized_support_coverage, \
    basic_statistical_measure_num, discriminator_measure_rf


class Evaluator:
    def __init__(self, real_dataset: Dataset):
        self.real_dataset = real_dataset
        self.evaluation_funcs: List[Callable] = []
        self.evaluation_funcs_kwargs: List[Dict] = []

    def register_evaluation_function(self,
                                     eval_func: Literal['log_corr', 'k_means', 'ml_cat', 'rsc', 'bsm', 'discriminator'],
                                     **kwargs) -> None:
        if eval_func == 'log_corr':
            self.evaluation_funcs.append(log_transformed_correlation_score)
            if 'categorical_cols' not in kwargs.keys():
                kwargs['categorical_cols'] = self.real_dataset.cat_cols
        elif eval_func == 'k_means':
            self.evaluation_funcs.append(k_means_score)
        elif eval_func == 'ml_cat':
            self.evaluation_funcs.append(ml_efficiency_cat)
            if 'relative' not in kwargs.keys():
                kwargs['relative'] = False
            if 'predict_col' not in kwargs.keys():
                kwargs['predict_col'] = self.real_dataset.predict_col
            if 'outcome_cols' not in kwargs.keys():
                kwargs['outcome_cols'] = self.real_dataset.outcome_cols
            if 'metric' not in kwargs.keys():
                raise AttributeError(f'For ml_cat the attribute "metric" needs to be provided.')
        elif eval_func == 'bsm':
            self.evaluation_funcs.append(basic_statistical_measure_num)
            if len(self.real_dataset.num_cols) == 0:
                raise ValueError(f'The provided dataset does not contain any numerical columns, bsm cannot be used '
                                 f'here.')
            if 'num_columns' not in kwargs.keys():
                kwargs['num_columns'] = self.real_dataset.num_cols
        elif eval_func == 'rsc':
            self.evaluation_funcs.append(regularized_support_coverage)
            if 'include_num' not in kwargs.keys():
                kwargs['include_num'] = False
            if 'categorical_cols' not in kwargs.keys():
                kwargs['categorical_cols'] = self.real_dataset.cat_cols
        elif eval_func == 'discriminator':
            self.evaluation_funcs.append(discriminator_measure_rf)
        else:
            raise AttributeError(f"eval_func need to be one of the following: ['log_corr', 'k_means', 'ml_cat', 'rsc', "
                                 f"'bsm', 'discriminator'], but {eval_func} was provided instead.")

        self.evaluation_funcs_kwargs.append(kwargs)

    def evaluate(self, df_synth: pd.DataFrame) -> float:
        if len(self.evaluation_funcs) == 0:
            raise ValueError("No evaluation functions provided. Ensure evaluation functions were registered through "
                             "'register_evaluation_function' before calling 'evaluate'.")
        sum_score = 0
        for func, func_kwargs in zip(self.evaluation_funcs, self.evaluation_funcs_kwargs):
            if func.__name__ == 'ml_efficiency_cat':
                sum_score += ml_efficiency_cat(orig_df_train=self.real_dataset.df_train,
                                               orig_df_test=self.real_dataset.df_test,
                                               synth_df_train=df_synth, **func_kwargs)
            else:
                sum_score += func(self.real_dataset.df_train, df_synth, **func_kwargs)

        return sum_score / len(self.evaluation_funcs)

