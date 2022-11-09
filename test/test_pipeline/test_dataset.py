from textwrap import dedent

import pandas as pd
import pytest

from ml_project.pipeline.dataset import (
    download_dataset,
    enrich_dataset
)

DATAFRAME_TEXT = dedent("""\
    header_1,header_2,header_3
    1,2,hello
    3,4,it
    1,4,is
    3,2,example
""")


@pytest.fixture()
def dataframe_fio(tmpdir):
    fio = tmpdir.join('dataframe.csv')
    fio.write(DATAFRAME_TEXT)

    return fio


class TestDownloadDataset:

    def test_easy_download_dataset(self, dataframe_fio):
        df = download_dataset(dataframe_fio)

        assert isinstance(df, pd.DataFrame)
        assert 4 == df.shape[0]
        assert 3 == df.shape[1]
        assert ['header_1', 'header_2', 'header_3'] == df.columns.tolist()


class TestEnrichDataset:

    def test_enrich_dataset(self):
        df = pd.DataFrame(data={
            'h1': [1, 2, 3, 4],
            'h2': [5, 6, 7, 8]
        })
        column = [10, 11, 12, 13]
        name = 'new_h'

        new_df = enrich_dataset(df, column, name, copy=True)

        assert df is not new_df
        assert df.columns.tolist() + [name] == new_df.columns.tolist()
        assert column == new_df[name].tolist()

        new_df_2 = enrich_dataset(df, column, name, copy=False)

        assert new_df_2 is df
