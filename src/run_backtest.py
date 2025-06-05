############################################################################
### QPMwP - RUN BACKTEST
############################################################################

import sys
import os
import pandas as pd
import numpy as np

# Add source directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Imports
from selection.custom_selection import CustomSelectionItemBuilder
from backtesting.backtest_item_builder_functions import bibfn_predicted_returns
from optimization.optimization_base import Optimization
from optimization.score_maximizer import ScoreMaximizer
from backtesting.backtest_service import BacktestService
from backtesting.backtest_item_builder_classes import SelectionItemBuilder, OptimizationItemBuilder
from backtesting.backtest_item_builder_functions import (
    bibfn_selection_high_roa,
    build_selection_vol_prof,
    bibfn_return_series,
    bibfn_bm_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
)

from backtesting.backtest_data import BacktestData

# -----------------------------
# Load your data
# -----------------------------
market_data = pd.read_parquet(f'/Users/elenetsaouse/Downloads/market_data.parquet')
jkp_data = pd.read_parquet('/Users/elenetsaouse/Downloads/jkp_data.parquet')

# Fill NaN values with 0.0 (or any other placeholder)
market_data.fillna(0.0, inplace=True)
jkp_data.fillna(0.0, inplace=True)

spi_index = pd.read_csv(
    "/Users/elenetsaouse/qpmwp-course/qpmwp-course/data/spi_index.csv",
    header=None,
    names=["date", "bm_series"]
)

# Add this right after imports
def debug_optimization(bs, date):
    """Helper function to print optimization debug info"""
    print(f"\n=== DEBUG {date} ===")
    print("[SELECTION]")
    print(f"Selected IDs: {bs.selection.selected}")
    print(f"# Selected: {len(bs.selection.selected)}")
    
    print("\n[OPTIMIZATION DATA]")
    for k, v in bs.optimization_data.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            print(f"{k}: {type(v)} | Shape: {v.shape}")
            print(v.head(3) if len(v) > 0 else "Empty")
        else:
            print(f"{k}: {type(v)} | Value: {v}")
    
    if hasattr(bs, 'optimization'):
        print("\n[OPTIMIZATION SETUP]")
        print(f"Objective type: {type(bs.optimization.objective)}")
        print(f"Constraint counts:")
        print(f"- Budget: {len(bs.optimization.constraints.budget)}")
        print(f"- Box: {len(bs.optimization.constraints.box)}")
        print(f"- Linear: {len(bs.optimization.constraints.linear)}")


def validate_and_cleanup_optimization_data(bs, date):
    print(f"[DEBUG] Validating optimization data on {date}")

    od = bs.optimization_data
    for key in ['return_series', 'scores', 'features']:
        if key in od:
            val = od[key]
            if isinstance(val, pd.DataFrame):
                if val.isnull().values.any():
                    print(f" ⚠️ NaNs in {key}. Filling with 0.")
                    val.fillna(0.0, inplace=True)  # Fill NaN with 0
                od[key] = val.astype(np.float64)
            elif isinstance(val, pd.Series):
                if val.isnull().any():
                    print(f" ⚠️ NaNs in {key}. Filling with 0.")
                    val.fillna(0.0, inplace=True)  # Fill NaN with 0
                od[key] = val.astype(np.float64)

# Convert 'date' to datetime and set as index
spi_index["date"] = pd.to_datetime(spi_index["date"], format="%d/%m/%Y")  # match your format
spi_index.set_index("date", inplace=True)

spi_index.sort_index(inplace=True)  # optional but useful

my_data = BacktestData()
my_data.market_data = market_data
my_data.jkp_data = jkp_data
my_data.bm_series = spi_index["bm_series"]  # Attach benchmark return series as attribute `bm_series`

# -----------------------------
# Define rebalancing dates
# -----------------------------
list_of_rebalancing_dates = sorted(
    market_data.index.get_level_values("date").unique()
)
list_of_rebalancing_dates = [
    d.strftime("%Y-%m-%d")
    for d in pd.to_datetime(list_of_rebalancing_dates)
    if d.day == 1
]

first_jkp_date = jkp_data.index.get_level_values("date").min()

list_of_rebalancing_dates = [
    d for d in list_of_rebalancing_dates
    if pd.to_datetime(d) > first_jkp_date + pd.Timedelta(days=365)
]

# Filter dates to only those with SPI index available
list_of_rebalancing_dates = [
    d for d in list_of_rebalancing_dates
    if pd.to_datetime(d) >= pd.to_datetime("1999-01-02")
]



# -----------------------------
# Define selection and optimization
# -----------------------------
selection_items = {
    'scores': SelectionItemBuilder(bibfn=bibfn_selection_high_roa, item_name='high_roa', filter_name='scores')
}

from backtesting.backtest_item_builder_functions import bibfn_predicted_returns

optimization_items = {
    'return_series': OptimizationItemBuilder(bibfn=bibfn_return_series, width=252),
    'bm_series': OptimizationItemBuilder(bibfn=bibfn_bm_series, width=252),
    'budget_constraint': OptimizationItemBuilder(
        bibfn=bibfn_budget_constraint,
        has_constraint_matrix=True  
    ),
    'box_constraints': OptimizationItemBuilder(bibfn=bibfn_box_constraints, lower=0, upper=0.1),
    'scores': OptimizationItemBuilder(bibfn=bibfn_predicted_returns, item_name='scores')
}


# -----------------------------
# Create and run BacktestService
# -----------------------------
bs = BacktestService(
    data=my_data,
    selection_item_builders=selection_items,
    optimization_item_builders=optimization_items,
    optimization=ScoreMaximizer(),
    settings={
        'rebdates': list_of_rebalancing_dates,
        'cost_fixed': 0.01,
        'cost_variable': 0.002,
    }
)

weights_dict = {}
prev_weights = None  # Initialize outside the backtest loop
portfolio_returns = []  # Store returns after costs
failed_dates = []  # To track failed optimization dates

for date in list_of_rebalancing_dates:
    print(f"\nProcessing: {date}")
    
    # 1. Prepare rebalancing first
    bs.prepare_rebalancing(date)
    
    
    # 2. Early validation checks (no duplicates)
    if not bs.selection.selected:
        print(f"⚠️ Skipping {date}: No selected assets")
        failed_dates.append(date)
        continue
        
    if len(bs.selection.selected) < 3:
        print(f"⚠️ Skipping {date}: Only {len(bs.selection.selected)} assets (need at least 3)")
        failed_dates.append(date)
        continue
        
    if ('return_series' not in bs.optimization_data or 
        bs.optimization_data['return_series'].isnull().all().all()):
        print(f"⚠️ Skipping {date}: Invalid return series")
        failed_dates.append(date)
        continue
    
    # 3. Debug output
    debug_optimization(bs, date)
    
    # 4. Handle features - moved before optimization
    features = bs.optimization_data.get("features")
    if features is None or features.empty:
        print(f"⚠️ Missing features for {date}. Initializing empty features.")
        features = pd.DataFrame(0, index=range(252), columns=bs.selection.selected)
    
    # Ensure feature alignment
    # Load from CSV used during model training
    expected_features_path = "/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/ml_dataset.csv"
    expected_features_df = pd.read_csv(expected_features_path)
    expected_features = expected_features_df.drop(columns=["fwd_1m_ret", "date", "id"], errors="ignore").columns.tolist()
    missing_features = list(set(expected_features) - set(features.columns))
    if missing_features:
        print(f"⚠️ Adding {len(missing_features)} missing features (filled with 0)")
        fill_df = pd.DataFrame(0, index=features.index, columns=missing_features)
        features = pd.concat([features, fill_df], axis=1)[expected_features]
    
    bs.optimization_data["features"] = features
    
    # 5. Handle scores
    scores = bs.optimization_data.get("scores")
    if scores is not None:
        if isinstance(scores, pd.Series):
            scores = scores.fillna(0).astype(np.float64)
            bs.optimization_data["scores"] = scores
    
    # 6. Set objective and validate
    bs.optimization.set_objective(bs.optimization_data)
    validate_and_cleanup_optimization_data(bs, date)
    
    # 7. Run optimization

    features = bs.optimization_data.get("features")
    scores = bs.optimization_data.get("scores")
    returns = bs.optimization_data.get("return_series")

    if (features is None or features.empty or
        scores is None or len(scores) < 3 or
        returns is None or returns.shape[1] < 3):
        print(f"⚠️ Skipping {date}: Not enough valid data for optimization")
        failed_dates.append(date)
        continue

    try:
        result = bs.optimization.solve()

    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        failed_dates.append(date)
        continue


    
    # 8. Process results
    weights = result.get("weights")
    weights = weights[weights > 0.01]
    weights /= weights.sum()



    if weights is None or len(weights) == 0:
        print("⚠️ No weights returned from optimization")
        failed_dates.append(date)
        continue
    
    print(f"✅ Successfully generated weights for {len(weights)} assets")
    weights_dict[date] = weights
    
    # 9. Calculate portfolio return
    returns = bs.optimization_data["return_series"]
    avg_returns = returns.mean()
    
    # Clean weights
    weights = (weights.dropna()
               .groupby(level=0).sum()
               .reindex(avg_returns.index, fill_value=0))
    
    portfolio_return = avg_returns @ weights
    
    # Calculate costs
    cost = 0.0
    if prev_weights is not None:
        turnover = (weights - prev_weights).abs().sum()
        cost = 0.002 * turnover
    
    fixed_cost = 0.01 / 12
    net_return = portfolio_return - cost - fixed_cost
    portfolio_returns.append(net_return)
    print(f"[INFO] Net return on {date}: {net_return:.6f}")

    prev_weights = weights

# Save weights
# Ensure all weights are Series with unique index (asset IDs)
for date, w in weights_dict.items():
    weights_dict[date] = pd.Series(w).groupby(level=0).sum()

weights_df = pd.DataFrame(weights_dict).T
weights_df.index.name = "date"
os.makedirs("output", exist_ok=True)
weights_df.to_csv("output/portfolio_weights.csv")
print("Portfolio weights saved to output/portfolio_weights.csv")

# Save returns


returns_df = pd.DataFrame({
    "date": list(weights_dict.keys()),
    "return": portfolio_returns
})
returns_df.to_csv("output/portfolio_returns.csv", index=False)
print("Portfolio returns saved to output/portfolio_returns.csv")


# Save failed dates (optional)
failed_dates_df = pd.DataFrame(failed_dates, columns=["Failed Dates"])
failed_dates_df.to_csv("output/failed_dates.csv")
print("Failed dates saved to output/failed_dates.csv")
