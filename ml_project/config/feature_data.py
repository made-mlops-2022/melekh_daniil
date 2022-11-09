from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureData:
    feature: List[str]
    target: Optional[str]
