import pickle

from sklearn.pipeline import Pipeline


def dump_pipeline(pipeline: Pipeline, path: str) -> None:
    """
    Dump test_pipeline

    :param pipeline: test_pipeline
    :param path: path to save/dump

    :return: test_pipeline
    """

    with open(path, 'wb') as fio:
        pickle.dump(pipeline, fio)


def load_pipeline(path: str) -> Pipeline:
    """
    Load test_pipeline

    :param path: path to load

    :return: test_pipeline
    """

    with open(path, 'rb') as fio:
        pipeline: Pipeline = pickle.load(fio)

    return pipeline
