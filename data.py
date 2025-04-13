import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional
import pandas as pd


@dataclass
class Dataset:
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    num_cols: List[str]
    cat_cols: List[str]
    outcome_cols: List[str]
    predict_col: Optional[str]


def get_dataset(name: Literal['bcw', 'actg', 'heart_uci', 'stroke', 'thyroid_cancer']) -> Dataset:
    if not os.path.exists(f'data/{name}'):
        raise FileNotFoundError(f"Error: The path 'data/{name}' does not exist.")

    df_train = pd.read_csv(f'data/{name}/train.csv')
    df_test = pd.read_csv(f'data/{name}/test.csv')

    with open(f'data/{name}/columns.json', 'r') as f:
        info_json = json.load(f)

    cat_cols = info_json['boolean'] + info_json['categorical']
    num_cols = info_json['integer'] + info_json['float']
    outcome_cols = info_json['outcome']
    predict_col = outcome_cols[0]
    dataset = Dataset(df_train, df_test, num_cols, cat_cols, outcome_cols, predict_col)
    return dataset


def get_synthetic_data(dataset_name: Literal['bcw', 'actg', 'heart_uci', 'stroke', 'thyroid_cancer']) -> pd.DataFrame:
    if not os.path.exists(f'data/{dataset_name}'):
        raise FileNotFoundError(f"Error: The path 'data/{dataset_name}' does not exist.")

    synthetic_dfs = []
    dir_path = f'data/{dataset_name}/synthetic_data'
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):
            synthetic_dfs.append(pd.read_csv(dir_path + f'/{file}'))
    df_synth = pd.concat(synthetic_dfs, ignore_index=True)

    if dataset_name == 'bcw':
        df_synth['Bare_Nuclei'] = df_synth['Bare_Nuclei'].apply(lambda x: str(x).strip())

    return df_synth






