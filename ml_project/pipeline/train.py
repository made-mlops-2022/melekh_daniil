from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from ml_project.config.train_data import TrainData

NAME_TO_MODEL = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'SVC': SVC
}


def get_model(train_data: TrainData) -> ClassifierMixin:
    """
    Train model

    :param train_data: config for fitting and init

    :return: model
    """

    class_model = NAME_TO_MODEL.get(train_data.model_name)

    if class_model is None:
        raise Exception(f'Not found model {class_model} from {NAME_TO_MODEL.keys()}')

    model = class_model(random_state=train_data.random_state)

    return model


def collect_pipeline(scaler: TransformerMixin,
                     model: ClassifierMixin) -> Pipeline:
    """
    Create test_pipeline

    :param scaler: scaler for test_pipeline
    :param model: model for test_pipeline

    :return:
    """

    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    return pipeline

