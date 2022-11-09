import pandas as pd

from ml_project.config.train_pipeline_data import TrainPipelineData


def dump_metric(train_pipeline_data: TrainPipelineData,
                metric_train: float,
                metric_test: float,
                metric_name: str) -> None:
    """
    Dump model metric

    :param train_pipeline_data: train test_pipeline object with specific param
    :param metric_train: metric for train
    :param metric_test: metric for test
    :param metric_name: metric name (e.x. F1, Recall, Precision etc)

    :return: None
    """

    df = pd.DataFrame(data={
        'metric_name': [metric_name],
        'train': [metric_train],
        'test': [metric_test]
    })

    df.to_csv(train_pipeline_data.metric_path, index=False)
