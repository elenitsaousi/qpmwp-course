import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load SPI benchmark data
spi_index = pd.read_csv("/Users/elenetsaouse/qpmwp-course/qpmwp-course/data/spi_index.csv", names=["date", "bm_series"], header=None)
spi_index["date"] = pd.to_datetime(spi_index["date"], format="%d/%m/%Y")
spi_index["bm_series"] = pd.to_numeric(spi_index["bm_series"], errors="coerce")

# Load your portfolio returns
portfolio_returns = pd.read_csv("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/portfolio_returns.csv")
portfolio_returns["date"] = pd.to_datetime(portfolio_returns["date"])


# Merge on date
df = pd.merge(portfolio_returns, spi_index, on="date", how="inner")
df = df.sort_values("date")
df = df.dropna(subset=["return", "bm_series"])
print(df[["return", "bm_series"]].isna().sum())


# Compute cumulative returns
df["cumulative_strategy"] = (1 + df["return"]).cumprod()
df["cumulative_spi"] = (1 + df["bm_series"]).cumprod()

# Compute rolling 3-year returns (36 months)
df["rolling_strategy"] = df["cumulative_strategy"].pct_change(periods=36)
df["rolling_spi"] = df["cumulative_spi"].pct_change(periods=36)

# Descriptive statistics
summary = {
    "Metric": ["Mean Return", "Std Dev", "Sharpe Ratio", "Max Drawdown"],
    "Strategy": [
        df["return"].mean(),
        df["return"].std(),
        df["return"].mean() / df["return"].std(),
        (df["cumulative_strategy"] / df["cumulative_strategy"].cummax() - 1).min()
    ],
    "SPI Index": [
        df["bm_series"].mean(),
        df["bm_series"].std(),
        df["bm_series"].mean() / df["bm_series"].std(),
        (df["cumulative_spi"] / df["cumulative_spi"].cummax() - 1).min()
    ]
}
summary_df = pd.DataFrame(summary)

# Plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["cumulative_strategy"], label="Strategy", linewidth=2)
plt.plot(df["date"], df["cumulative_spi"], label="SPI Index", linewidth=2)
plt.title("Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/cumulative_returns.png")

# Plot rolling 3-year returns
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["rolling_strategy"], label="Strategy", linestyle='--')
plt.plot(df["date"], df["rolling_spi"], label="SPI Index", linestyle='--')
plt.title("Rolling 3-Year Returns")
plt.xlabel("Date")
plt.ylabel("Rolling Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/rolling_returns.png")

print("Performance Summary:")
print(summary_df)




cum = df["cumulative_strategy"]
drawdown = cum / cum.cummax() - 1

plt.figure(figsize=(10, 4))
plt.plot(df["date"], drawdown, color="red")
plt.title("Drawdown of Strategy")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.savefig("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/drawdown_curve.png")
