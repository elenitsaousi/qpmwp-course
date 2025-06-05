from dataclasses import dataclass
import pandas as pd

@dataclass
class SelectionItem:
    universe: pd.Index
