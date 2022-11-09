"""
Data object
"""

from dataclasses import dataclass

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass()
class SplitData:
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    shuffle: bool = True
