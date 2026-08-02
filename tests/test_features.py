import pandas as pd


def load_data():
    return pd.read_parquet(
        "data/processed/processed_v0_v1.parquet"
    )


def test_rolling_avg_10():

    df = load_data()

    assert "rolling_avg_10" in df.columns
    assert df["rolling_avg_10"].notnull().all()
    assert (df["rolling_avg_10"] > 0).all()


def test_volume_sum_10():

    df = load_data()

    assert "volume_sum_10" in df.columns
    assert df["volume_sum_10"].notnull().all()
    assert (df["volume_sum_10"] >= 0).all()