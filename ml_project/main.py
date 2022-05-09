import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


import hydra
from hydra.core.config_store import ConfigStore

from src.preprocessing.cat_mte import MTE
from config import ModelConfig

cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelConfig)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: ModelConfig):
    """pipeline build & fit"""
    # data loading
    tr_data = pd.read_csv(f"{cfg.paths.data}/{cfg.files.train_data}")
    te_data = pd.read_csv(f"{cfg.paths.data}/{cfg.files.test_data}")
    tr_target = pd.read_csv(f"{cfg.paths.data}/{cfg.files.train_target}").iloc[:, 0]
    te_target = pd.read_csv(f"{cfg.paths.data}/{cfg.files.test_target}").iloc[:, 0]

    # preprocessing and model fitting
    pipeline = Pipeline(
        [
            ("preprocessor_mte", MTE()),
            (
                "model_lr",
                LogisticRegression(
                    penalty=cfg.model_params.penalty,
                    C=cfg.model_params.C,
                    solver=cfg.model_params.solver,
                    max_iter=cfg.model_params.max_iter,
                ),
            ),
        ]
    )
    pipeline.fit(
        X=tr_data,
        y=tr_target,
    )


if __name__ == "__main__":
    main()
