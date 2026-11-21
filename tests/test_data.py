import pandas as pd
import pytest

from credit_xai import config
from credit_xai.data import age_group, clean, load_raw, split


def test_load_raw_shape(raw_file):
    df = load_raw(raw_file)
    assert df.shape == (200, len(config.RAW_COLUMNS))


def test_clean_decodes_everything(raw_file):
    df = clean(load_raw(raw_file))
    assert "personal_status_sex" not in df.columns
    assert "class" not in df.columns
    assert set(df["sex"]) <= {"M", "F"}
    assert set(df[config.TARGET]) <= {0, 1}
    # nessun codice "Axx" deve sopravvivere
    for col in config.CODEBOOK:
        assert not df[col].str.match(r"^A\d+$").any(), col
    assert df.isna().sum().sum() == 0


def test_target_mapping(raw_file):
    raw = load_raw(raw_file)
    df = clean(raw)
    assert df[config.TARGET].sum() == (raw["class"] == 2).sum()


def test_clean_rejects_unknown_codes(raw_file):
    raw = load_raw(raw_file)
    raw.loc[0, "purpose"] = "A999"
    with pytest.raises(ValueError, match="non riconosciuti"):
        clean(raw)


def test_split_is_stratified(raw_file):
    df = clean(load_raw(raw_file))
    train, test = split(df)
    assert len(train) + len(test) == len(df)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.05


def test_age_group():
    groups = age_group(pd.Series([20, 25, 26, 40, 70]))
    assert list(groups) == ["<=25", "<=25", "26-35", "36-50", ">50"]
