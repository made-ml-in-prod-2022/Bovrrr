from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    data: str


@dataclass
class Files:
    file_name: str


@dataclass
class Params:
    penalty: str
    C: float
    solver: str
    max_iter: int


@dataclass
class ModelConfig:
    paths: Paths
    files: Files
    params: Params
