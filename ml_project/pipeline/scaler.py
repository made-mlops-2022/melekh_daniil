from sklearn.preprocessing import (
    Normalizer, StandardScaler, MinMaxScaler, RobustScaler
)

from ml_project.config.scaler_data import ScalerData

NAME_TO_SCALER = {
    'Normalizer': Normalizer,
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'RobustScaler': RobustScaler
}


def get_scaler(scaler_data: ScalerData) -> any:
    """
    Transformer by feature data

    :param scaler_data: feature data
                         see ml_project.config.feature_data.FeatureData

    :return: transformer from scikit-learn
    """

    class_scaler = NAME_TO_SCALER.get(scaler_data.scaler_name)

    if class_scaler is None:
        raise Exception(f'Not found scaler {class_scaler} from {NAME_TO_SCALER.keys()}')

    scaler = class_scaler()

    return scaler
