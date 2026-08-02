from pathlib import Path
import pandas as pd


def process_stock(csv_file):
    """Process one stock CSV exactly like the reference notebook."""

    df = pd.read_csv(csv_file)

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Stock name
    df["stock_name"] = csv_file.stem.split("__")[0]

    # Sort (README says not to assume sorted data)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward fill missing values
    df = df.ffill()

    # Timestamp index for rolling windows
    df = df.set_index("timestamp")

    # Last 10 minutes
    df["rolling_avg_10"] = (
        df["close"]
        .rolling(window="10min", min_periods=1)
        .mean()
    )

    df["volume_sum_10"] = (
        df["volume"]
        .rolling(window="10min", min_periods=1)
        .sum()
    )

    # Restore timestamp column
    df = df.reset_index()

    # Target: price after 5 minutes
    future_close = df["close"].shift(-5)
    df["target"] = (future_close > df["close"]).astype(int)

    # Remove rows without future target
    df = df.iloc[:-5]

    return df


def process_folder(folder):
    dfs = []

    for csv_file in sorted(Path(folder).glob("*.csv")):
        print(f"Processing {csv_file.name}")
        dfs.append(process_stock(csv_file))

    return pd.concat(dfs, ignore_index=True)


def main():

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Iteration 1 (v0)
    # ----------------------------
    v0 = process_folder("data/v0")

    v0.to_parquet(
        output_dir / "processed_v0.parquet",
        index=False,
    )

    print("\nIteration 1")
    print(v0.shape)

    # ----------------------------
    # Iteration 2 (v0 + v1)
    # ----------------------------
    v1 = process_folder("data/v1")

    merged = pd.concat([v0, v1], ignore_index=True)

    merged.to_parquet(
        output_dir / "processed_v0_v1.parquet",
        index=False,
    )

    print("\nIteration 2")
    print(merged.shape)

    print("\nFinished preprocessing.")


if __name__ == "__main__":
    main()