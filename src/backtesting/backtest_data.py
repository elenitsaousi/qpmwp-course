############################################################################
### QPMwP - CLASS BacktestData
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------

# Standard library imports
from typing import Optional
import warnings

# Third party imports
import pandas as pd


class BacktestData():

    def __init__(self):
        pass

    def get_return_series(self, ids: Optional[pd.Series] = None, end_date: Optional[str] = None,
                      width: Optional[int] = None, fillna_value: Optional[float] = 0.0) -> pd.DataFrame:
    
        # Default behavior if no data is provided
        X = self.market_data.pivot_table(
            index='date',
            columns='id',
            values='price'
        )

        # Default values for ids, end_date, and width if not provided
        if ids is None:
            ids = X.columns

        if end_date is None:
            end_date = X.index.max().strftime('%Y-%m-%d')

        if width is None:
            width = X.shape[0] - 1

        # Filter valid data based on the end date and selected IDs
        X = X[X.index <= end_date][ids].tail(width + 1).pct_change(fill_method=None).iloc[1:]

        # Handle NaN values by filling them with fillna_value (ensuring fillna_value is not None)
        if X.isnull().values.any():
            print(f"[ERROR] NaN values detected in return series for {end_date}.")
            if fillna_value is None:
                print("[WARNING] fillna_value is None. Using default 0.0.")
                fillna_value = 0.0  # Use default fillna_value if None is passed
            X.fillna(fillna_value, inplace=True)

        # If the return series is empty after filling NaNs
        if X.empty:
            print(f"[ERROR] No valid data found for {end_date}.")
            return pd.DataFrame()  # Return empty DataFrame if no data

        return X  # Return the cleaned DataFrame



    def get_volume_series(self,
                          ids: Optional[pd.Series] = None,
                          end_date: Optional[str] = None,
                          width: Optional[int] = None,
    ) -> pd.DataFrame:

        X = self.market_data.pivot_table(
            index='date',
            columns='id',
            values='liquidity',
        )
        if ids is None:
            ids = X.columns
        if end_date is None:
            end_date = X.index.max().strftime('%Y-%m-%d')
        if width is None:
            width = X.shape[0]
        return X[X.index <= end_date][ids].tail(width)

    def get_characteristic_series(self,
                                  field: str,
                                  ids: Optional[pd.Series] = None,
                                  end_date: Optional[str] = None,
                                  width: Optional[int] = None,
    ) -> pd.DataFrame:

        X = self.jkp_data.pivot_table(
            index='date',
            columns='id',
            values=field,
        )
        if ids is None:
            ids = X.columns
        if end_date is None:
            end_date = X.index.max().strftime('%Y-%m-%d')
        if width is None:
            width = X.shape[0]
        return X[X.index <= end_date][ids].tail(width)
