import pandas as pd

class OptimizationResults:
    def __init__(self, weights: pd.Series, status: str = "unknown"):
        self.weights = weights
        self.status = status
