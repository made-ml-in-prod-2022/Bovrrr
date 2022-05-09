from dataclasses import dataclass


@dataclass
class Paths:
    """Data class for dir paths"""

    log: str
    raw_data: str
    data: str


@dataclass
class Files:
    """Data class for data-file names"""

    train_data: str
    test_data: str
    train_target: str
    test_target: str


@dataclass
class ModelParams:
    """Data class for model fitting params"""

    model_name: str
    penalty: str
    C: float
    solver: str
    max_iter: int
    n_jobs: int


@dataclass
class TrTeSplitParams:
    """Data class for data preprocessing"""

    file_name: str
    target_column_name: str
    random_state: int
    shuffle: bool
    test_size: float


@dataclass
class ModelConfig:
    """Data class for whole model fitting"""

    paths: Paths
    files: Files
    model_params: ModelParams
    train_test_split_params: TrTeSplitParams
