from dataclasses import dataclass

import yaml

from marshmallow_dataclass import class_schema

from ml_project.config.feature_data import FeatureData
from ml_project.config.scaler_data import ScalerData
from ml_project.config.split_data import SplitData
from ml_project.config.train_data import TrainData


@dataclass()
class TrainPipelineData:
    input_dataframe_path: str
    output_model_path: str
    metric_path: str
    scaler_data: ScalerData
    feature_data: FeatureData
    split_data: SplitData
    train_data: TrainData


TrainPipelineDataSchema = class_schema(TrainPipelineData)


def get_train_pipeline_data(path: str) -> TrainPipelineData:
    """
    Read data/param for train test_pipeline

    :param path: path to *.yaml file

    :return: data object as TrainPipelineData
    """

    with open(path, 'r') as fio:
        schema = TrainPipelineDataSchema()

        return schema.load(yaml.safe_load(fio))
