import pandas as pd
import numpy as np
import os

# Load data
weights = pd.read_csv("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/portfolio_weights.csv", index_col="date", parse_dates=True)
market_data = pd.read_parquet(f'/Users/elenetsaouse/Downloads/market_data.parquet')

# Compute daily returns
# Compute daily returns
market_data.sort_index(inplace=True)
market_data["ret"] = market_data.groupby("id")["price"].pct_change()

# Drop duplicate (date, id) index entries
market_data = market_data[~market_data.index.duplicated()]

# Unstack to (date × id)
returns = market_data["ret"].unstack("id")
returns.index = pd.to_datetime(returns.index)


# Simulation
monthly_returns = []
prev_weights = None
fixed_cost = 0.000833  # ~1% annual
tx_cost_rate = 0.002   # 0.2% transaction cost

for date in weights.index:
    month_ret = returns[returns.index.to_period('M') == pd.Period(date, 'M')]
    if month_ret.empty:
        continue

    w = weights.loc[date]
    w = w / w.sum()
    valid_assets = w.index.intersection(month_ret.columns)
    w = w[valid_assets]
    month_ret = month_ret[valid_assets]

    # Portfolio return
    port_ret = month_ret.dot(w)
    gross_return = (1 + port_ret).prod() - 1

    # Transaction cost
    if prev_weights is not None:
        aligned_prev = prev_weights.reindex(w.index).fillna(0)
        turnover = np.abs(w - aligned_prev).sum()
        tx_cost = turnover * tx_cost_rate
    else:
        tx_cost = 0

    net_return = gross_return - fixed_cost - tx_cost
    monthly_returns.append([date, net_return])

    prev_weights = w.copy()

# Results
result_df = pd.DataFrame(monthly_returns, columns=["date", "net_return"]).set_index("date")
result_df["cumulative"] = (1 + result_df["net_return"]).cumprod()

# Save
os.makedirs("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output", exist_ok=True)
result_df.to_csv("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/portfolio_performance.csv")
print("Saved to /Users/elenetsaouse/qpmwp-course/qpmwp-course/output/portfolio_performance.csv")