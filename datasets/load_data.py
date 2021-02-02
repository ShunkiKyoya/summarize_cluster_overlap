"""Function to Load Datasets for Real Data Experiment.

Refer to README.md for the required procedures to apply this function.
"""

import io
import os

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pyper


def load_data(data_name, *, scaling=True):
    """Load Data.

    Args:
        data_name (str): the name of the data.
        scaling (bool): whether to scale or not.
    Returns:
        Tuple[ndarray, ndarray]: X, Z_true.
    """
    path_here = os.path.dirname(__file__)

    if data_name == 'AIS':
        r = pyper.R(use_pandas=True)
        r('library(locfit)')
        r('data(ais)')
        r('df_r <- ais')
        df = r.get('df_r')

        X = df[[' BMI ', ' LBM ', ' BFat ']].values
        Z_true = (df['sex'].replace({b'female': 0, b'male': 1})).values

    elif data_name == 'beetles':
        r = pyper.R(use_pandas=True)
        r('library(fdm2id)')
        r('data(beetles)')
        df = r.get('beetles')

        X = df.loc[:, [' Width ', ' Angle ']].values
        Z_true = df['Species'].replace(
            {b'concinna': 0, b'heikertingeri': 1, b'heptapotamica': 2}
        ).values

    elif data_name == 'crabs':
        r = pyper.R(use_pandas=True)
        r('library(MASS)')
        r('data(crabs)')
        r('df_r <- crabs')
        df = r.get('df_r')

        X = df[[' FL ', ' RW ', ' CL ', ' CW ', ' BD ']].values
        Z_true = df[' index '].copy()
        Z_true[(df['sp'] == b'B') & (df['sex'] == b'M')] = 0
        Z_true[(df['sp'] == b'B') & (df['sex'] == b'F')] = 1
        Z_true[(df['sp'] == b'O') & (df['sex'] == b'M')] = 2
        Z_true[(df['sp'] == b'O') & (df['sex'] == b'F')] = 3
        Z_true = Z_true.values

    elif data_name == 'DLBCL':
        r = pyper.R(use_pandas=True)
        r('library(EMMIXuskew)')
        r('data(DLBCL)')
        X = r.get('DLBCL').values
        Z_true = r.get('true.clusters')
        X = X[Z_true != 0]
        Z_true = Z_true[Z_true != 0]

    elif data_name == 'ecoli':
        # downloaded from https://archive.ics.uci.edu/ml/datasets/ecoli
        with open('{}/ecoli.data'.format(path_here), 'r') as f:
            df_txt = f.read()
            df = pd.read_csv(io.StringIO(df_txt), sep=r'\s+', header=None)

            X = df.loc[:, 1:8].copy()
            X = X[
                (df[8] == 'cp') |
                (df[8] == 'im') |
                (df[8] == 'imU') |
                (df[8] == 'om') |
                (df[8] == 'pp')
            ].values
            X = X[:, [0, 1, 4, 5, 6]]

            Z_true = df[8].copy()
            Z_true = Z_true[
                (df[8] == 'cp') |
                (df[8] == 'im') |
                (df[8] == 'imU') |
                (df[8] == 'om') |
                (df[8] == 'pp')
            ]
            Z_true = Z_true.replace(
                {'cp': 0, 'im': 1, 'imU': 2, 'om': 3, 'pp': 4}
            ).values

    elif data_name == 'seeds':
        # downloaded from https://archive.ics.uci.edu/ml/datasets/seeds
        with open('{}/seeds_dataset.txt'.format(path_here), 'r') as f:
            df = pd.read_table(f, sep=r'\s+', header=None)

            X = df.iloc[:, 0:7].values
            Z_true = df.iloc[:, 7].values - 1

    elif data_name == 'yeast':
        # downloaded from http://archive.ics.uci.edu/ml/datasets/yeast
        with open('{}/yeast.data'.format(path_here), 'r') as f:
            df_txt = f.read()
            df = pd.read_csv(io.StringIO(df_txt), sep=r'\s+', header=None)
        df_sub = df[(df[9] == 'CYT') | (df[9] == 'ME3')]

        X = df_sub[[1, 3, 7]].values  # 1, 3, 7
        Z_true = df_sub[9].replace({'CYT': 0, 'ME3': 1}).values.astype(int)

    elif data_name == 'wisconsin':
        X, Z_true = load_breast_cancer(return_X_y=True)
        X = X[:, [23, 24, 1]]

    X = X.astype('float64')

    if scaling:
        scaler = StandardScaler(copy=True)
        X = scaler.fit_transform(X)

    return X, Z_true
