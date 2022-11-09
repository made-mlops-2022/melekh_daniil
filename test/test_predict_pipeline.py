from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ml_project.config.train_pipeline_data import get_train_pipeline_data
from ml_project.pipeline.dataset import download_dataset
from ml_project.pipeline.scaler import get_scaler
from ml_project.pipeline.train import get_model, collect_pipeline
from ml_project.pipeline.train_test_split import train_test_split_with_config, dataframe_to_x_y

PATH_TRAIN_PIPELINE_DATA = 'test/data/test_train_config.yaml'

COLLECT_COLUMN = ['age', 'sex', 'cp', 'trestbps',
                  'chol', 'fbs', 'restecg', 'thalach',
                  'exang', 'oldpeak', 'slope',
                  'ca', 'thal', 'condition']

COLLECT_INPUT_COLUMN = [
    'age', 'sex', 'cp', 'trestbps',
    'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope',
    'ca', 'thal'
]

TARGET = 'condition'


class TestTrainPipeline:

    def test_train_pipeline(self):
        train_data = get_train_pipeline_data(PATH_TRAIN_PIPELINE_DATA)

        dataframe = download_dataset(train_data.input_dataframe_path)

        assert 8 == dataframe.shape[0]
        assert len(COLLECT_COLUMN) == dataframe.shape[1]
        assert COLLECT_COLUMN == dataframe.columns.tolist()

        df_train, df_test = train_test_split_with_config(
            dataframe, train_data.split_data
        )

        assert train_data.split_data.test_size < 0.5
        assert df_train.shape[0] > df_test.shape[0]
        assert df_train.shape[1] == df_test.shape[1]
        assert df_train.columns.tolist() == df_test.columns.tolist()

        x_train, y_train = dataframe_to_x_y(df_train, train_data.feature_data)

        assert COLLECT_INPUT_COLUMN == x_train.columns.tolist()
        assert list(y_train) == df_train[TARGET].tolist()

        x_test, y_test = dataframe_to_x_y(df_test, train_data.feature_data)

        assert COLLECT_INPUT_COLUMN == x_test.columns.tolist()
        assert list(y_test) == df_test[TARGET].tolist()

        scaler = get_scaler(train_data.scaler_data)

        assert isinstance(scaler, StandardScaler)

        model = get_model(train_data.train_data)

        assert isinstance(model, LogisticRegression)

        pipeline = collect_pipeline(scaler, model)

        assert isinstance(pipeline, Pipeline)

        pipeline.fit(x_train, y_train)

        predict_train = pipeline.predict(x_train)
        predict_test = pipeline.predict(x_test)

        assert len(y_train) == len(predict_train)
        assert len(y_test) == len(predict_test)

        metric_train = f1_score(y_train, predict_train)
        metric_test = f1_score(y_test, predict_test)

        assert 0 <= metric_train <= 1
        assert 0 <= metric_test <= 1
