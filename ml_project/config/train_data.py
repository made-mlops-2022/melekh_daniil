from dataclasses import dataclass

RANDOM_STATE = 42
DEFAULT_MODEL = 'LogisticRegression'


@dataclass()
class TrainData:
    model_name: str = DEFAULT_MODEL
    random_state: int = RANDOM_STATE
