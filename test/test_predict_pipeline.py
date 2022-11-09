from sklearn.pipeline import Pipeline

from ml_project.config.predict_pipeline_data import get_predict_pipeline_data
from ml_project.pipeline.dataset import download_dataset, enrich_dataset
from ml_project.pipeline.dump_load_pipeline import load_pipeline
from ml_project.pipeline.train_test_split import dataframe_to_x_y

PATH_TRAIN_PIPELINE_DATA = 'test/data/test_predict_config.yaml'

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

TARGET = 'test_predict_condition'


class TestPredictPipeline:

    def test_train_pipeline(self):
        predict_data = get_predict_pipeline_data(PATH_TRAIN_PIPELINE_DATA)

        dataframe = download_dataset(predict_data.input_dataset_path)

        assert 8 == dataframe.shape[0]
        assert len(COLLECT_COLUMN) == dataframe.shape[1]
        assert COLLECT_COLUMN == dataframe.columns.tolist()

        pipeline = load_pipeline(predict_data.model_path)

        assert isinstance(pipeline, Pipeline)

        x_dataframe, _ = dataframe_to_x_y(dataframe,
                                          predict_data.feature_data)

        assert COLLECT_INPUT_COLUMN == x_dataframe.columns.tolist()

        y_predict = pipeline.predict(x_dataframe)

        assert len(y_predict) == dataframe.shape[0]

        enriched_dataframe = enrich_dataset(
            dataframe,
            y_predict,
            predict_data.output_target_field
        )

        assert TARGET in enriched_dataframe.columns
