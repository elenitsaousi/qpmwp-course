############################################################################
### QPMwP - BACKTEST ITEM BUILDER CLASSES
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from abc import ABC, abstractmethod
from typing import Any

# Third party imports
import numpy as np
import pandas as pd





class BacktestItemBuilder(ABC):
    '''
    Base class for building backtest items.
    This class should be inherited by specific item builders.
    
    # Arguments
    kwargs: Keyword arguments that fill the 'arguments' attribute.
    '''

    def __init__(self, **kwargs):
        self._arguments = {}
        self._arguments.update(kwargs)

    @property
    def arguments(self) -> dict[str, Any]:
        return self._arguments

    @arguments.setter
    def arguments(self, value: dict[str, Any]) -> None:
        self._arguments = value

    @abstractmethod
    def __call__(self, service, rebdate: str) -> None:
        raise NotImplementedError("Method '__call__' must be implemented in derived class.")



class SelectionItemBuilder(BacktestItemBuilder):
    '''
    Callable Class for building selection items in a backtest.
    '''

    def __call__(self, bs: 'BacktestService', rebdate: str) -> None:
        '''
        Build selection item from a custom function.

        :param bs: The backtest service.
        :param rebdate: The rebalance date.
        :raises ValueError: If 'bibfn' is not defined or not callable.
        '''

        selection_item_builder_fn = self.arguments.get('bibfn')
        if selection_item_builder_fn is None or not callable(selection_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        item_value = selection_item_builder_fn(bs = bs, rebdate = rebdate, **self.arguments)
        item_name = self.arguments.get('item_name')

        # Add selection item
        bs.selection.add_filtered(filter_name = item_name, value = item_value)
        return None



class OptimizationItemBuilder(BacktestItemBuilder):
    '''
    Callable Class for building optimization data items in a backtest.
    '''

    def __init__(self, has_constraint_matrix: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.has_constraint_matrix = has_constraint_matrix

    def __call__(self, bs, rebdate: str) -> None:
        optimization_item_builder_fn = self.arguments.get('bibfn')
        if optimization_item_builder_fn is None or not callable(optimization_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        # Filter out 'bibfn' from the arguments
        filtered_args = {k: v for k, v in self.arguments.items() if k != 'bibfn'}

        result = optimization_item_builder_fn(bs=bs, rebdate=rebdate, **filtered_args)

        # --- FIX START ---
        if isinstance(result, tuple):
            print(f"[DEBUG] bibfn returned tuple: {type(result)}, unpacking...")
            result = result[0]  # or handle accordingly if your function returns (dict, something_else)
        # --- FIX END ---

        if self.has_constraint_matrix:
            if not isinstance(result, dict):
                raise TypeError(f"[ERROR] Expected dict from bibfn, but got {type(result)}")

            A, b = result.get("A"), result.get("b")
            G, h = result.get("G"), result.get("h")
            if A is not None and b is not None:
                print(f"[DEBUG] Injecting constraint matrix A: A.shape = {A.shape}, b.shape = {b.shape}")
                bs.optimization.constraints.add_matrix(A=A, b=b)
            if G is not None and h is not None:
                print(f"[DEBUG] Injecting box constraint G: G.shape = {G.shape}, h.shape = {h.shape}")
                bs.optimization.constraints.add_matrix(G=G, h=h)

        else:
            item_name = self.arguments.get('item_name')
            if result is not None and item_name is not None:
                bs.optimization_data[item_name] = result

        return None

    
