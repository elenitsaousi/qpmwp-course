import numpy as np
import pandas as pd

def build_features(market_data, jkp_data):
    df = pd.merge(market_data, jkp_data, on=["id", "date"], how="inner")
    df["log_mktcap"] = np.log1p(df["mktcap"])
    df["liquidity_rank"] = df["liquidity"].rank()
    return df
