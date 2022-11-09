from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.config.feature_data import FeatureData
from ml_project.config.split_data import SplitData


def train_test_split_with_config(dataframe: pd.DataFrame,
                                 split_data: SplitData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe on train and test
    by params from config

    :param dataframe: dataframe
    :param split_data: object with config param for splitting
                       see ml_project.config.split_data.SplitData

    :return: train dataframe and test dataframe
    """

    df_train, df_test = train_test_split(
        dataframe,
        test_size=split_data.test_size,
        random_state=split_data.random_state,
        shuffle=split_data.shuffle,
    )

    return df_train, df_test


def dataframe_to_x_y(dataframe: pd.DataFrame,
                     feature_data: FeatureData) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Get X (input features) an Y (target) from dataframe
    using

    :param dataframe: dataframe
    :param feature_data: object with config param for splitting
                         see ml_project.config.feature_data.FeatureData

    :return: X (input features) an Y (target) from dataframe
    """

    df_input, target = dataframe[feature_data.feature], None

    if feature_data.target is not None:
        target = dataframe[feature_data.target]

    return df_input, target
