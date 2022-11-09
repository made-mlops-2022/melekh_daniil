import pandas as pd


def dump_dataframe(dataframe: pd.DataFrame, path: str) -> None:
    """
    Dump dataframe

    :param dataframe: dataframe
    :param path: path to save/dump

    :return: None
    """

    dataframe.to_csv(path, index=False)
