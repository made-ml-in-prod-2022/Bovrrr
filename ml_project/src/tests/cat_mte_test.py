import pandas as pd
from ..preprocessing.cat_mte import MTE


CAT_FEAT_NUM = 8
DATA_PATH = "../../data/raw_data"
DATA_FILE_NAME = "heart_cleveland_upload.csv"
TARGET = "condition"


def test_mte_instance():
    mte_obj = MTE()
    assert isinstance(mte_obj, MTE)


def test_mte_cat_feat_num():
    mte_obj = MTE()
    data = pd.read_csv(f"{DATA_PATH}/{DATA_FILE_NAME}").drop(
        [TARGET],
        axis=1,
        inplace=True,
    )
    mte_obj.fit(data)
    assert len(mte_obj.cols) == 8
