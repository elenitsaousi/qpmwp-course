import pandas as pd
import numpy as np

# Load data
market_data = pd.read_parquet("/Users/elenetsaouse/Downloads/market_data.parquet").reset_index()
jkp_data = pd.read_parquet("/Users/elenetsaouse/Downloads/jkp_data.parquet").reset_index()

# === Feature Engineering ===
def build_features(market_data: pd.DataFrame, jkp_data: pd.DataFrame) -> pd.DataFrame:
    market_data = market_data.sort_values(["id", "date"])
    jkp_data = jkp_data.sort_values(["id", "date"])

    # Technical indicators
    market_data["ret_1d"] = market_data.groupby("id")["price"].pct_change()
    market_data["ret_5d"] = market_data.groupby("id")["price"].pct_change(5)
    market_data["ret_21d"] = market_data.groupby("id")["price"].pct_change(21)
    market_data["momentum_3m"] = market_data.groupby("id")["price"].pct_change(63)
    market_data["momentum_12m"] = market_data.groupby("id")["price"].pct_change(252)
    market_data["vol_21d"] = market_data.groupby("id")["ret_1d"].transform(lambda x: x.rolling(21).std())
    market_data["vol_63d"] = market_data.groupby("id")["ret_1d"].transform(lambda x: x.rolling(63).std())
    market_data["liq_rank"] = market_data.groupby("date")["liquidity"].rank(pct=True)

    # Fundamental ratios
    jkp_data["book_to_price"] = 1 / jkp_data.get("pe_exi", pd.Series(np.nan))
    jkp_data["ebit_ev"] = jkp_data.get("ebit", 0.0) / (jkp_data.get("ev", 1.0) + 1e-6)
    jkp_data["roa"] = jkp_data.get("niq", 0.0) / (jkp_data.get("at", 1.0) + 1e-6)
    jkp_data["accruals"] = (
        (jkp_data.get("act", 0.0) - jkp_data.get("che", 0.0)) -
        (jkp_data.get("lct", 0.0) - jkp_data.get("dlc", 0.0))
    )

    for col in ["book_to_price", "ebit_ev", "roa", "accruals"]:
        jkp_data[f"z_{col}"] = jkp_data.groupby("date")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

    # Target: 1-month forward return
    market_data = market_data[market_data["price"].notnull()].copy()
    market_data["price_fwd"] = market_data.groupby("id")["price"].shift(-21)
    market_data["fwd_1m_ret"] = (market_data["price_fwd"] / market_data["price"]) - 1

    # Align
    market_data["date"] = pd.to_datetime(market_data["date"])
    jkp_data["date"] = pd.to_datetime(jkp_data["date"])
    market_data["id"] = market_data["id"].astype(str)
    jkp_data["id"] = jkp_data["id"].astype(str)

    df = pd.merge(
        market_data[["date", "id", "ret_5d", "ret_21d", "momentum_3m", "momentum_12m", "vol_21d", "vol_63d", "liq_rank", "fwd_1m_ret"]],
        jkp_data[["date", "id", "z_book_to_price", "z_ebit_ev", "z_roa", "z_accruals"]],
        on=["date", "id"],
        how="inner"
    )

    df = df.dropna(subset=["fwd_1m_ret"])
    df = df[df["fwd_1m_ret"].between(-1, 1)]  # cap extreme returns
    print("Final sample:", df.shape)

    return df


# Run & Save
dataset = build_features(market_data, jkp_data)
dataset.to_csv("output/ml_dataset.csv", index=False)
print("✅ Dataset saved.")
