from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


import hydra
from hydra.core.config_store import ConfigStore
from ml_project.src.preprocessing import cat_mte

from src.preprocessing.cat_mte import MTE
from ml_project.config import ModelConfig

cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelConfig)


def main(cfg: ModelConfig):
    # Model
    model = LogisticRegression()
    preprocessor = MTE()


if __name__ == "__main__":
    main()
