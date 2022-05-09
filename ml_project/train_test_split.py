import pandas as pd
from sklearn.model_selection import train_test_split

import hydra
from hydra.core.config_store import ConfigStore

from config import ModelConfig

cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelConfig)

FROM_HYDRA_OUTPUT = "../../../"


@hydra.main(config_path="conf", config_name="config")
def main(cfg: ModelConfig):
    def dump_csv(data: pd.core.frame.DataFrame, path: str, file_name):
        data.to_csv(
            path_or_buf=f"{FROM_HYDRA_OUTPUT}/{path}/{file_name}",
            index=False,
        )

    raw_data = pd.read_csv(
        f"{FROM_HYDRA_OUTPUT}/{cfg.paths.raw_data}/{cfg.train_test_split_params.file_name}"
    )
    target = raw_data[[cfg.train_test_split_params.target_column_name]].copy()
    raw_data.drop(
        [cfg.train_test_split_params.target_column_name],
        axis=1,
        inplace=True,
    )

    tr_data, te_data, tr_target, te_target = train_test_split(
        raw_data,
        target,
        shuffle=cfg.train_test_split_params.shuffle,
        random_state=cfg.train_test_split_params.random_state,
        test_size=cfg.train_test_split_params.test_size,
        stratify=target,
    )

    del raw_data, target

    dump_csv(
        tr_data,
        cfg.paths.data,
        cfg.files.train_data,
    )
    dump_csv(
        te_data,
        cfg.paths.data,
        cfg.files.test_data,
    )
    dump_csv(
        tr_target,
        cfg.paths.data,
        cfg.files.train_target,
    )
    dump_csv(
        te_target,
        cfg.paths.data,
        cfg.files.test_target,
    )


if __name__ == "__main__":
    main()
