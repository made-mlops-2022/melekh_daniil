from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureData:
    # scaler: str
    feature: List[str]
    target: Optional[str]
