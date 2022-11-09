import click
import pandas as pd

from ml_project.config.predict_pipeline_data import (
    get_predict_pipeline_data,
    PredictPipelineData
)
from ml_project.log.logger import get_logger
from ml_project.pipeline.dataset import download_dataset, enrich_dataset
from ml_project.pipeline.dump_load_pipeline import load_pipeline
from ml_project.pipeline.dump_predict import dump_dataframe
from ml_project.pipeline.train_test_split import dataframe_to_x_y

logger = get_logger('PredictPipeline')


@click.command()
@click.argument("config_path")
def cli_predict(config_path: str) -> None:
    """
    Run training using CLI

    :param config_path: path to *.yaml config
                        see .config.predict_pipeline_data.PredictPipelineData

    :return: None
    """

    logger.info('Cli train test_pipeline')
    predict_pipeline_data = get_predict_pipeline_data(config_path)
    logger.info('Yaml-file uploaded successfully')

    predict(predict_pipeline_data)


def predict(predict_pipeline_data: PredictPipelineData) -> pd.DataFrame:
    """
    Predict by model

    :param predict_pipeline_data: config for predict
                                  see .config.train_pipeline_data.PredictPipelineData

    :return: dataframe
    """

    logger.info('Start train test_pipeline')

    dataframe = download_dataset(predict_pipeline_data.input_dataset_path)
    logger.info('Loaded dataframe with shape: %s' % str(dataframe.shape))

    pipeline = load_pipeline(predict_pipeline_data.model_path)
    logger.info('Pipeline collected successfully')

    x_dataframe, _ = dataframe_to_x_y(dataframe,
                                      predict_pipeline_data.feature_data)
    logger.info('Got X_TRAIN successfully '
                'from DF_TRAIN: %s' % str(x_dataframe.shape))

    y_predict = pipeline.predict(x_dataframe)
    logger.info('Pipeline collected successfully')

    logger.info('Dataframe (before enriching) with columns %s'
                % list(dataframe.columns))
    enriched_dataframe = enrich_dataset(
        dataframe,
        y_predict,
        predict_pipeline_data.output_target_field
    )
    logger.info('Enriched dataframe with columns %s'
                % list(enriched_dataframe.columns))

    dump_dataframe(
        enriched_dataframe,
        predict_pipeline_data.output_predict_path
    )
    logger.info('Saved dataframe by path: %s' %
                predict_pipeline_data.output_predict_path)

    return enriched_dataframe


if __name__ == "__main__":
    cli_predict()
