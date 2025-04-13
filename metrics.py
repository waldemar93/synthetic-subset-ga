import copy
from typing import Optional, List, Literal
import numpy as np
import pandas as pd
from dython.nominal import _comp_assoc
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from catboost import CatBoostClassifier


def basic_statistical_measure_num(orig_df: DataFrame, synth_df: DataFrame, num_columns: List[str] = None,
                                  clip_val: int = 1):
    """
    Compute a similarity score between two datasets based on basic statistical measures
    (mean, median, and standard deviation) for numerical columns.

    Parameters:
        orig_df (DataFrame): The original dataset used as a reference.
        synth_df (DataFrame): The synthetic dataset to compare against the original.
        num_columns (List[str], optional): List of numerical column names to evaluate.
                                           If None, all numerical columns in `orig_df` are used.
        clip_val (int, optional): Maximum allowed difference for each metric. Values beyond this
                                  threshold are clipped. Defaults to 1.

    Returns:
        float: The averaged similarity score across mean, median, and standard deviation
               differences, ranging from 0 (worst) to 1 (best).
    """

    if num_columns is None:
        num_columns = list(orig_df.select_dtypes(include=[int, float]))

    # compute the relative difference between real and synth dataset for mean, median and std
    mean_dif = abs((orig_df[num_columns].mean() - synth_df[num_columns].mean()) / orig_df[num_columns].mean())
    median_dif = abs((orig_df[num_columns].median() - synth_df[num_columns].median()) / orig_df[num_columns].median())
    std_dif = abs((orig_df[num_columns].std() - synth_df[num_columns].std()) / orig_df[num_columns].std())

    if clip_val is not None:
        mean_dif.clip(upper=clip_val, inplace=True)
        median_dif.clip(upper=clip_val, inplace=True)
        std_dif.clip(upper=clip_val, inplace=True)

    median_dif = [0 if pd.isna(v) else v for v in median_dif]

    # compute the scores for mean, median and std
    mean_score = 1 - (sum(mean_dif) / len(mean_dif))
    median_score = 1 - (sum(median_dif) / len(median_dif))
    std_score = 1 - (sum(std_dif) / len(std_dif))
    # best 1, worst 0, final score
    basic_measure = (mean_score + median_score + std_score) / 3

    return basic_measure


def log_transformed_correlation_score(orig_df: DataFrame, synth_df: DataFrame,
                                      categorical_cols: Optional[List[str]] = None, clip_val: Optional[int] = 1) \
        -> float:
    """
    Compute a similarity score based on log-transformed correlation matrices between two datasets.

    This function calculates correlation matrices for the original and synthetic datasets using:
        - Pearson's correlation for continuous-continuous pairs,
        - Correlation Ratio for categorical-continuous pairs,
        - Theil's U for categorical-categorical pairs.

    It applies a log transformation to the absolute correlation values (with zeros replaced by a small epsilon),
    preserves the original sign, computes relative errors between the transformed matrices, and optionally clips them.
    The final score is `1 - mean(relative error)`, where a higher score (max 1) indicates a better match.

    Note:
        This method may not perform well when correlations are near-perfect (close to ±1).

    Parameters:
        orig_df (DataFrame): The original dataset used as a reference.
        synth_df (DataFrame): The synthetic dataset to compare against the original.
        categorical_cols (List[str], optional): List of column names to treat as categorical.
                                                If None, 'auto' is used to infer them.
        clip_val (int, optional): Maximum allowed value for relative errors.
                                  If None, no clipping is applied. Defaults to 1.

    Returns:
        float: The log-transformed correlation overlap score between the datasets, ranging from 0 (no similarity)
               to 1 (perfect match).
    """

    if categorical_cols is None:
        categorical_cols = 'auto'

    orig_corr = _comp_assoc(orig_df, categorical_cols, False, theil_u=True, clustering=False, bias_correction=True,
                            nan_strategy='ignore', nan_replace_value=-1)[0]
    synth_corr = _comp_assoc(synth_df, categorical_cols, False, theil_u=True, clustering=False, bias_correction=True,
                             nan_strategy='ignore', nan_replace_value=-1)[0]
    for col in orig_corr.columns:
        orig_corr[col] = orig_corr[col].astype(float)
        synth_corr[col] = synth_corr[col].astype(float)

    # Signs for the correlation matrices
    sign_orig_corr = np.sign(orig_corr)
    sign_synth_corr = np.sign(synth_corr)

    epsilon = np.finfo(float).eps  # Smallest positive float
    log_orig_corr = np.log(np.maximum(abs(orig_corr), epsilon))
    log_synth_corr = np.log(np.maximum(abs(synth_corr), epsilon))

    # Relative errors in correlation matrices
    rel_errors = abs((((sign_orig_corr * log_orig_corr) - (sign_synth_corr * log_synth_corr)) /
                      (sign_orig_corr * log_orig_corr)))

    # Clip the relative errors to a maximum value
    if clip_val is not None:
        rel_errors = rel_errors.clip(upper=clip_val)

    # calculate the mean, 1 highest value, 0 lowest value
    score = 1 - rel_errors.mean().mean()

    return score


def regularized_support_coverage(orig_df: DataFrame, synth_df: DataFrame, categorical_cols: List[str],
                                 clip_ratio: float = 1, clip_col: int = 1, include_num: bool = True,
                                 num_bins: int = 10) -> float:
    """
    Compute the regularized support coverage metric to evaluate the similarity between original
    and synthetic datasets.

    This metric compares the distributional support (i.e., presence and frequency of values or
    bins) of each feature in the original and synthetic datasets. It evaluates categorical
    columns by comparing value counts, and optionally includes numerical columns by binning
    them into intervals. Ratios are scaled by dataset sizes and clipped for robustness. The
    final score is the mean coverage across all considered columns.

    Parameters:
        orig_df (DataFrame): The original dataset used as a reference.
        synth_df (DataFrame): The synthetic dataset to compare against the original.
        categorical_cols (List[str]): List of column names treated as categorical.
        clip_ratio (float, optional): Maximum allowed value for per-bin/category ratio between
                                      synthetic and original data. Defaults to 1.
        clip_col (int, optional): Maximum allowed support coverage per column. Defaults to 1.
        include_num (bool, optional): Whether to include numerical columns in the metric.
                                      Defaults to True.
        num_bins (int, optional): Number of bins used to discretize numerical columns.
                                  Defaults to 10.

    Returns:
        float: The regularized support coverage score ranging from 0 (poor coverage) to 1 (perfect coverage).
    """

    # depending on the size of synthetic data
    scaling_factor = len(orig_df) / len(synth_df)

    # sum of all support coverage metrics
    sum_support_coverage = 0

    for col in categorical_cols:
        # for each categorical column get the count of each value for real and synthetic data
        orig_val_counts = orig_df[col].value_counts(sort=False, dropna=False)
        synth_val_counts = synth_df[col].value_counts(sort=False, dropna=False)

        # the number of unique values in the real data
        n = len(orig_val_counts.keys())
        support_cov = 0

        for key in orig_val_counts.keys():
            # if value not in synthetic data, it counts as 0
            if key not in synth_val_counts.keys():
                continue

            # determine the ratio of synthetic and real data samples coverage, use clip_ratio as maximum
            # because of special int type, nan values are always part of this, even if there are 0 values
            if orig_val_counts[key] == 0:
                n = n - 1
                continue
            ratio = (synth_val_counts[key] / orig_val_counts[key]) * scaling_factor

            if ratio > clip_ratio:
                ratio = clip_ratio

            support_cov += ratio

        # calculate the support coverage for a single variable
        support_cov /= n

        # if the support coverage is higher than clip_col, clip it to this value
        if support_cov > clip_col:
            support_cov = clip_col

        sum_support_coverage += support_cov

    if include_num:
        # all columns that are not categorical are considered numerical
        num_cols = [col for col in orig_df.columns if col not in categorical_cols]

        for col in num_cols:
            # get the cutoff values for num_bins for each numerical columns
            cut_offs = pd.qcut(orig_df[col], q=num_bins, retbins=True, duplicates='drop')[1]
            support_cov = 0

            for i in range(len(cut_offs)-1):
                # get the min value and max value for each cutoff
                min_val = cut_offs[i]
                max_val = cut_offs[i + 1]

                # closed buckets on the left and right
                # in case of the last bucket the maximum value is inclusive, otherwise not
                # count for each bucket how many real and synthetic samples there are
                if (i + 1) == len(cut_offs)-1:
                    orig_bucket_count = len(orig_df[(orig_df[col] >= min_val) & (orig_df[col] <= max_val)])
                    synth_bucket_count = len(synth_df[(synth_df[col] >= min_val) & (synth_df[col] <= max_val)])
                else:
                    orig_bucket_count = len(orig_df[(orig_df[col] >= min_val) & (orig_df[col] < max_val)])
                    synth_bucket_count = len(synth_df[(synth_df[col] >= min_val) & (synth_df[col] < max_val)])

                # calculate the ratio of synthetic datapoints to real datapoints
                ratio = (synth_bucket_count / orig_bucket_count) * scaling_factor

                # clip the ratio if it is higher than clip_ratio
                if ratio > clip_ratio:
                    ratio = clip_ratio

                support_cov += ratio

            # calculate support coverage for a singe numerical variable
            support_cov /= (len(cut_offs)-1)

            # if the support coverage exceeds clip_col, then set it to clip_col
            if support_cov > clip_col:
                support_cov = clip_col

            sum_support_coverage += support_cov

        # divide the sum of all support coverage values with the number of all columns (since all are used) for the
        # final metric
        reg_support_cov = sum_support_coverage / len(orig_df.columns)
    else:
        # divide the sum of all support coverage values with the number of all categorical columns for the final metric
        reg_support_cov = sum_support_coverage / len(categorical_cols)

    return reg_support_cov


def discriminator_measure_rf(orig_df: DataFrame, synth_df: DataFrame, test_ratio: float = 0.2) -> float:
    """
    Compute a discrimination score using a Random Forest classifier to evaluate how easily real
    and synthetic data can be distinguished.

    This function trains a Random Forest on a labeled dataset combining original and synthetic data.
    The model predicts whether a sample is real or synthetic. The closer the model’s accuracy is
    to random guessing (i.e., 0.5), the better the synthetic data mimics the real data.

    Features are preprocessed using label encoding for categorical variables, and NaNs are filled
    with a constant placeholder. Accuracy below 0.5 is adjusted to represent ideal indistinguishability.

    Parameters:
        orig_df (DataFrame): The original dataset.
        synth_df (DataFrame): The synthetic dataset.
        test_ratio (float, optional): Proportion of samples used for testing. Defaults to 0.2.

    Returns:
        float: A discrimination score between 0 (easily distinguishable) and 1 (indistinguishable).
    """

    # copy the data
    orig_df = copy.deepcopy(orig_df)
    synth_df = copy.deepcopy(synth_df)

    if len(synth_df) > len(orig_df):
        synth_df = synth_df.sample(n=len(orig_df), random_state=42).reset_index(drop=True)
    elif len(orig_df) > len(synth_df):
        orig_df = orig_df.sample(n=len(synth_df), random_state=42).reset_index(drop=True)

    # preprocessing
    # select non-numerical columns (ignoring booleans)
    non_numeric_cols = orig_df.select_dtypes(exclude=[int, float]).columns

    df = pd.concat([orig_df, synth_df], axis=0)
    for col in non_numeric_cols:
        # using LabelEncoder instead of OneHotEncoder because random forest is used
        encoder = LabelEncoder()
        # fit the encoder to original data
        encoder.fit(df[col])
        # transform real and synthetic data column-wise
        orig_df[col] = encoder.transform(orig_df[col])
        synth_df[col] = encoder.transform(synth_df[col])

    # determine columns containing NaN values
    nan_columns = orig_df.columns[orig_df.isna().any()].tolist()

    if len(nan_columns) > 0:
        # replace all NaN values with -9999999999 for all variables (categorical and numerical)
        orig_df.fillna(-9999999999, inplace=True)
        synth_df.fillna(-9999999999, inplace=True)

    # add extra column to show which of the datapoints is real and which is not
    orig_df['real_point'] = 1
    synth_df['real_point'] = 0

    # split real and synth data into train and test, size depends on the provided test_ratio
    orig_df_train, orig_df_test, synth_df_train, synth_df_test = train_test_split(orig_df, synth_df, random_state=42,
                                                                                  test_size=test_ratio)

    # concatenate real and synth train dataframes into one dataframe
    df_train = pd.concat([orig_df_train, synth_df_train], ignore_index=True, sort=False)
    # concatenate real and synth test dataframes into one dataframe
    df_test = pd.concat([orig_df_test, synth_df_test], ignore_index=True, sort=False)

    # shuffle train dataset
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # get y for train and test data
    y_train = df_train['real_point']
    y_test = df_test['real_point']

    # remove y from X train and test
    df_train.drop(['real_point'], axis=1, inplace=True)
    df_test.drop(['real_point'], axis=1, inplace=True)

    # initiate a RandomForestClassifier with default parameters
    rf_model = RandomForestClassifier(random_state=42)
    # fit the model on the training data
    rf_model.fit(df_train, y_train)
    # predict on the test data
    y_pred = rf_model.predict(df_test)
    # calculate the accuracy for the test data
    accuracy = accuracy_score(y_test, y_pred)

    # best case 0.5, if it is lower than that, it means that the classifier cannot distinguish between real and synth
    # datapoint, therefore, we set the value to the ideal 0.5
    if accuracy < 0.5:
        accuracy = 0.5

    # 1 best, 0 worst
    score = 1 - (accuracy - 0.5) * 2
    return score


def k_means_score(orig_df: DataFrame, synth_df: DataFrame, k: int = 10, clip_ratio: float = 1,
                  cluster_on_real_only=True):
    """
    Compute a clustering-based similarity score using the k-means algorithm.

    This function applies k-means clustering either to the original data alone or to the combined
    original and synthetic data. It then evaluates how well the synthetic data covers the
    clusters derived from the original dataset. The score reflects the average ratio of synthetic
    to real data points within each cluster, clipped to a maximum value.

    Categorical and boolean features are one-hot encoded, and numerical features are scaled.
    Missing values are handled with imputation and separate scaling strategies.

    Parameters:
        orig_df (DataFrame): The original dataset used for reference.
        synth_df (DataFrame): The synthetic dataset to be evaluated.
        k (int, optional): Number of clusters for k-means. Defaults to 10.
        clip_ratio (float, optional): Maximum allowed ratio of synthetic to real points in a cluster.
                                      Defaults to 1.
        cluster_on_real_only (bool, optional): If True, clusters are learned from real data only.
                                               If False, real and synthetic data are clustered together.
                                               Defaults to True.

    Returns:
        float: The k-means coverage score, ranging from 0 (poor match) to `clip_ratio` (perfect match).
    """

    # depending on the size of synthetic data
    scaling_factor = len(orig_df) / len(synth_df)

    # copy data
    orig_df = copy.deepcopy(orig_df)
    synth_df = copy.deepcopy(synth_df)

    # get different types of columns
    num_cols = [col for col in orig_df.select_dtypes(include=[int, float]).columns if len(orig_df[col].unique()) > 3]
    non_numeric_cols = list(orig_df.select_dtypes(exclude=[int, float]).columns)
    bool_missing_cols = [col for col in orig_df.select_dtypes(include=[int, float]).columns
                         if len(orig_df[col].unique()) == 3]
    non_numeric_cols += bool_missing_cols

    # special handling of columns containing NaN
    nan_columns = orig_df.columns[orig_df.isna().any()].tolist()
    num_nan_cols = []
    if len(nan_columns) > 0:
        num_nan_cols = [col for col in orig_df.select_dtypes(include=[int, float]).columns if col in nan_columns]
        cat_nan_cols = [col for col in nan_columns if col not in num_nan_cols]

        # replace NaN value with a very large numerical value, so it can be perceived as an outlier
        for col in num_nan_cols:
            orig_df[col].fillna(-9999999999, inplace=True)
            synth_df[col].fillna(-9999999999, inplace=True)

        for col in cat_nan_cols:
            orig_df[col].fillna('empty', inplace=True)
            synth_df[col].fillna('empty', inplace=True)

        # boolean values will be handled with OneHotEncoder
        num_nan_cols = [col for col in num_nan_cols if col not in non_numeric_cols]

    if len(non_numeric_cols) > 0:
        # use OneHotEncoder for categorical (and boolean with missing values) columns
        oe_scaler = OneHotEncoder()
        oe_scaler.fit(pd.concat([orig_df, synth_df], axis=0)[non_numeric_cols])

        train_cat = oe_scaler.transform(orig_df[non_numeric_cols]).toarray()
        orig_df.drop(non_numeric_cols, axis=1, inplace=True)

        synth_cat = oe_scaler.transform(synth_df[non_numeric_cols]).toarray()
        synth_df.drop(non_numeric_cols, axis=1, inplace=True)
    else:
        train_cat = None
        synth_cat = None

    train_num_nan = None
    synth_num_nan = None

    # since NaN values are replaced with large numbers, these columns are encoded with RobustScaler
    if len(num_nan_cols) > 0:
        # remove the columns with NaN values from num_cols
        num_cols = [col for col in num_cols if col not in num_nan_cols]

        rs_scaler = RobustScaler()
        rs_scaler.fit(orig_df[num_nan_cols])

        train_num_nan = rs_scaler.transform(orig_df[num_nan_cols])
        orig_df.drop(num_nan_cols, axis=1, inplace=True)

        synth_num_nan = rs_scaler.transform(synth_df[num_nan_cols])
        synth_df.drop(num_nan_cols, axis=1, inplace=True)

    if len(num_cols) > 0:
        # use StandardScaler for all non NaN numerical columns
        std_scaler = StandardScaler()
        std_scaler.fit(orig_df[num_cols])

        train_num = std_scaler.transform(orig_df[num_cols])
        orig_df.drop(num_cols, axis=1, inplace=True)

        synth_num = std_scaler.transform(synth_df[num_cols])
        synth_df.drop(num_cols, axis=1, inplace=True)
    else:
        train_num = None
        synth_num = None

    # combine all transformed features to a numpy array for real and synthetic data

    arrays_to_concatenate = [arr for arr in [orig_df.to_numpy(), train_cat, train_num, train_num_nan] if arr is not None]
    arrays_to_concatenate_synth = [arr for arr in [synth_df.to_numpy(), synth_cat, synth_num, synth_num_nan] if arr is not None]

    real = np.concatenate(arrays_to_concatenate, axis=1)
    synth = np.concatenate(arrays_to_concatenate_synth, axis=1)

    # initiate KMeans model with k clusters
    model = KMeans(n_clusters=k, n_init=10, random_state=42)

    if cluster_on_real_only:
        # fit model to the real data
        model.fit(real)
    else:
        # fit model to all data
        all = np.concatenate([real, synth], axis=0)
        model.fit(all)

    # get the labels for each datapoint for the real and synthetic data
    real_clusters = list(model.predict(real))
    synth_clusters = list(model.predict(synth))

    # sum of the support coverage metric for all the clusters
    sum_val = 0
    # for each cluster determine the ratio of synthetic and real data samples coverage, use clip_ratio as maximum
    for i in range(k):
        real_count = real_clusters.count(i)
        synth_count = synth_clusters.count(i)

        if real_count > 0:
            ratio = (synth_count / real_count) * scaling_factor
        else:
            ratio = 0

        if ratio > clip_ratio:
            ratio = clip_ratio

        sum_val += ratio

    # total score is the sum of all the ratios (to a maximum of clip_ratio), divided by the number of clusters
    # higher number is better, 0 worst
    score = sum_val / k
    return score


def ml_efficiency_cat(orig_df_train: DataFrame, orig_df_test: DataFrame, synth_df_train: DataFrame, predict_col: str,
                      metric: Literal['f1', 'accuracy', 'roc_auc', 'f1_macro', 'f1_micro', 'mcc'], relative: bool,
                      outcome_cols: List[str]) -> float:
    """
    Evaluate how effectively a machine learning model trained on synthetic data can predict outcomes on real test data
    using CatBoostClassifier.

    This function compares model performance trained on synthetic data to that of a model trained on real data.
    It supports various evaluation metrics and optionally returns the relative efficiency (synthetic score divided
    by real score, clipped at 1).

    Categorical features are label encoded, and missing values are replaced with a constant. Outcome columns other
    than the target are removed to avoid leakage.

    Parameters:
        orig_df_train (DataFrame): Training set with original data.
        orig_df_test (DataFrame): Test set with original data.
        synth_df_train (DataFrame): Synthetic training set of the same structure as the original.
        predict_col (str): Name of the target variable column.
        metric (str): Metric used for evaluation. Options: 'f1', 'accuracy', 'roc_auc', 'f1_macro',
                      'f1_micro', 'mcc'.
        relative (bool): If True, return the score relative to a model trained on original data
                         (clipped at 1 if synthetic outperforms original).
        outcome_cols (List[str]): List of outcome columns; only the target column is retained.

    Returns:
        float: The evaluation score (absolute or relative) of the model trained on synthetic data.
    """

    # copy the data
    orig_df_train = copy.deepcopy(orig_df_train)
    orig_df_test = copy.deepcopy(orig_df_test)
    synth_df_train = copy.deepcopy(synth_df_train)

    # remove outcome cols
    columns_to_remove = [col for col in outcome_cols if col != predict_col]
    orig_df_train.drop(columns_to_remove, axis=1, inplace=True)
    orig_df_test.drop(columns_to_remove, axis=1, inplace=True)
    synth_df_train.drop(columns_to_remove, axis=1, inplace=True)

    # needed for the fitting of encoders
    orig_df = pd.concat([orig_df_train, orig_df_test], ignore_index=True, sort=False)

    # preprocessing
    # select non-numerical columns (ignoring booleans)
    non_numeric_cols = orig_df.select_dtypes(exclude=[int, float]).columns

    for col in non_numeric_cols:
        # using LabelEncoder instead of OneHotEncoder
        encoder = LabelEncoder()
        # fit the encoder to original data
        encoder.fit(orig_df[col])
        # transform real and synthetic data column-wise
        orig_df_train[col] = encoder.transform(orig_df_train[col])
        orig_df_test[col] = encoder.transform(orig_df_test[col])
        synth_df_train[col] = encoder.transform(synth_df_train[col])
        orig_df.drop([col], axis=1, inplace=True)

    # determine columns containing NaN values
    nan_columns = orig_df.columns[orig_df.isna().any()].tolist()

    if len(nan_columns) > 0:
        # replace all NaN values with -9999999999 for all variables (categorical and numerical)
        orig_df_train.fillna(-9999999999, inplace=True)
        orig_df_test.fillna(-9999999999, inplace=True)
        synth_df_train.fillna(-9999999999, inplace=True)

    # get y for train and test data
    orig_y_train = orig_df_train[predict_col]
    orig_y_test = orig_df_test[predict_col]
    synth_y_train = synth_df_train[predict_col]

    # remove y
    orig_df_train.drop([predict_col], axis=1, inplace=True)
    orig_df_test.drop([predict_col], axis=1, inplace=True)
    synth_df_train.drop([predict_col], axis=1, inplace=True)

    # initiate a CatBoostClassifier with provided parameters for synthetic data
    gb_model_synth = CatBoostClassifier(verbose=False, allow_writing_files=False, random_state=42)

    # fit the model on the synthetic data
    if len(synth_y_train.unique()) == 1:
        # to not get an error we change the last y to the missing category
        # get the missing item
        missing_items = set(orig_y_train.unique()) - set(synth_y_train.unique())
        if len(missing_items) == 1:
            synth_y_train = synth_y_train.copy()
            synth_y_train.iloc[-1] = list(missing_items)[0]

    gb_model_synth.fit(synth_df_train, synth_y_train)

    # predict on the test data
    y_pred_synth = gb_model_synth.predict(orig_df_test)

    # calculate the chosen metric for the test data
    if metric == 'f1':
        score_synth = f1_score(orig_y_test, y_pred_synth)
    elif metric == 'f1_micro':
        score_synth = f1_score(orig_y_test, y_pred_synth, average='micro')
    elif metric == 'f1_macro':
        score_synth = f1_score(orig_y_test, y_pred_synth, average='macro')
    elif metric == 'roc_auc':
        score_synth = roc_auc_score(orig_y_test, gb_model_synth.predict_proba(orig_df_test)[:, 1], average='micro')
    elif metric == 'accuracy':
        score_synth = accuracy_score(orig_y_test, y_pred_synth)
    elif metric == 'mcc':
        score_synth = matthews_corrcoef(orig_y_test, y_pred_synth)
    else:
        raise AttributeError(f'The evaluation metric must be one of the following: '
                             f'{", ".join(["f1", "mcc",  "accuracy", "roc_auc", "f1_macro", "f1_micro"])}, '
                             f'but {metric} was provided')

    if relative:
        gb_model_orig = CatBoostClassifier(verbose=False, allow_writing_files=False, random_state=42)
        gb_model_orig.fit(orig_df_train, orig_y_train)
        y_pred_orig = gb_model_orig.predict(orig_df_test)

        if metric == 'f1':
            score_orig = f1_score(orig_y_test, y_pred_orig)
        elif metric == 'f1_micro':
            score_orig = f1_score(orig_y_test, y_pred_orig, average='micro')
        elif metric == 'f1_macro':
            score_orig = f1_score(orig_y_test, y_pred_orig, average='macro')
        elif metric == 'roc_auc':
            score_orig = roc_auc_score(orig_y_test, y_pred_orig)
        elif metric == 'accuracy':
            score_orig = accuracy_score(orig_y_test, y_pred_orig)
        elif metric == 'mcc':
            score_synth = matthews_corrcoef(orig_y_test, y_pred_synth)

        # relative score
        score = score_synth / score_orig
        # in case synthetic score is better than the original one
        if score > 1:
            score = 1

        return score

    # otherwise return the actual score
    return score_synth
