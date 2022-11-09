from dataclasses import dataclass

import yaml

from marshmallow_dataclass import class_schema

from ml_project.config.feature_data import FeatureData


@dataclass
class PredictPipelineData:
    input_dataset_path: str
    model_path: str
    output_predict_path: str
    output_target_field: str
    feature_data: FeatureData


PredictPipelineDataSchema = class_schema(PredictPipelineData)


def get_predict_pipeline_data(path: str) -> PredictPipelineData:
    """
    Read data/param for predict using trained test_pipeline

    :param path: path to *.yaml file

    :return: data object as PredictPipelineData
    """

    with open(path, 'r') as fio:
        schema = PredictPipelineDataSchema()

        return schema.load(yaml.safe_load(fio))
