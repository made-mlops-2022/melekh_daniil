from typing import Union

import pandas as pd


def download_dataset(path: str) -> pd.DataFrame:
    """
    Download dataset

    :param path: path to *.csv dataset

    :return: DataFrame
    """

    return pd.read_csv(path)


def enrich_dataset(dataframe: pd.DataFrame,
                   new_column: Union[list, pd.Series],
                   name_new_column: str,
                   copy: bool = True) -> pd.DataFrame:
    """
    Enrich dataset: add column in dataset

    :param dataframe: dataframe
    :param new_column: new column for dataframe
    :param name_new_column: name for new column
    :param copy: if copy's True, returned dataframe is copy

    :return: enriched dataframe
    """

    df = dataframe

    if copy:
        df = dataframe.copy()

    df[name_new_column] = new_column

    return df
