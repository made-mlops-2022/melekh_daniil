"""
Main script: predict
"""

import click
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from config.train_pipeline_data import (
    get_train_pipeline_data,
    TrainPipelineData
)
from ml_project.log.logger import get_logger
from ml_project.pipeline.dump_load_pipeline import dump_pipeline
from ml_project.pipeline.metric import dump_metric
from ml_project.pipeline.scaler import get_scaler
from ml_project.pipeline.train import get_model, collect_pipeline
from pipeline.dataset import download_dataset
from pipeline.train_test_split import (
    train_test_split_with_config,
    dataframe_to_x_y
)

F1_METRIC = 'F1'

logger = get_logger('TrainPipeline')


@click.command()
@click.argument("config_path")
def cli_train_pipeline(config_path: str) -> None:
    """
    Run training using CLI

    :param config_path: path to *.yaml config
                        see .config.train_pipeline_data.TrainPipelineData

    :return: None
    """

    logger.info('Cli train test_pipeline')
    train_pipeline_data = get_train_pipeline_data(config_path)
    logger.info('Yaml-file uploaded successfully')

    train_pipeline(train_pipeline_data)


def train_pipeline(train_data: TrainPipelineData) -> Pipeline:
    """
    Train test_pipeline by config

    :param train_data: config for train
                       see .config.predict_pipeline_data.PredictPipelineData

    :return: test_pipeline
    """

    logger.info('Start train test_pipeline')

    dataframe = download_dataset(train_data.input_dataframe_path)
    logger.info('Loaded dataframe with shape: %s' % str(dataframe.shape))

    df_train, df_test = train_test_split_with_config(
        dataframe, train_data.split_data
    )
    logger.info('Splited train/test: %s/%s' % (df_train.shape[0], df_test.shape[0]))

    x_train, y_train = dataframe_to_x_y(df_train, train_data.feature_data)
    logger.info('Got X_TRAIN, Y_TRAIN successfully '
                'from DF_TRAIN: %s , %s' % (str(x_train.shape), str(y_train.shape)))

    x_test, y_test = dataframe_to_x_y(df_test, train_data.feature_data)
    logger.info('Got X_TEST, Y_TEST successfully '
                'from DF_TEST: %s , %s' % (str(x_test.shape), str(y_test.shape)))

    scaler = get_scaler(train_data.scaler_data)
    logger.info('Scaler loaded successfully')

    model = get_model(train_data.train_data)
    logger.info('Model loaded successfully')

    pipeline = collect_pipeline(scaler, model)
    logger.info('Pipeline collected successfully')

    logger.info('Start fit of model')
    pipeline.fit(x_train, y_train)
    logger.info('Finish fit of model: %s' % pipeline)
    logger.info('Param of fitted model: %s' % pipeline.get_params())

    predict_train = pipeline.predict(x_train)
    logger.info('Predicted objects: %s' % len(predict_train))

    predict_test = pipeline.predict(x_test)
    logger.info('Predicted objects: %s' % len(predict_test))

    metric_train = f1_score(y_train, predict_train)
    logger.info('F1 score of train: %s' % metric_train)

    metric_test = f1_score(y_test, predict_test)
    logger.info('F1 score of test: %s' % metric_test)

    dump_metric(train_data, metric_train, metric_test, F1_METRIC)
    logger.info('Saved metric by path: %s' % train_data.metric_path)

    dump_pipeline(pipeline, train_data.output_model_path)
    logger.info('Saved test_pipeline by path: %s' % train_data.output_model_path)

    logger.info('Finish train test_pipeline')

    return pipeline


if __name__ == "__main__":
    cli_train_pipeline()
