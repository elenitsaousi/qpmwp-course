import sys
import os
import pandas as pd
import numpy as np
import joblib
from cvxopt import matrix, solvers
from optimization.optimization_base import Optimization
from optimization.results import OptimizationResults
from optimization.constraints import Constraints


class ScoreMaximizer(Optimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = Constraints()
        self.model = joblib.load("/Users/elenetsaouse/qpmwp-course/qpmwp-course/output/ml_model.joblib")


    def set_objective(self, data: dict[str, pd.DataFrame]) -> None:
        """Define the objective function using ML predictions"""
        features = data.get("features")
        if features is None or features.empty:
            raise ValueError("Missing or empty features")

        # Ensure feature name compatibility
        features.columns = features.columns.astype(str)
        model_features = [f for f in self.model.feature_names_in_ if f in features.columns]
        selected_features = features[model_features].fillna(0.0)

        # Predict and transform scores
        raw_preds = np.nan_to_num(self.model.predict(selected_features))
        ranked_preds = pd.Series(raw_preds, index=features.index).rank(pct=True)

        sigmoid = lambda x: 1 / (1 + np.exp(-10 * (x - 0.5)))
        transformed_preds = ranked_preds.map(sigmoid)

        self.params["objective"] = transformed_preds
        self.params["x_init"] = pd.Series(1 / len(features), index=features.index)


    def solve(self) -> dict:
        """Solve the optimization problem using linear programming"""
        mu = self.params["objective"]
        n = len(mu)

        # Objective: maximize mu^T x → minimize -mu^T x
        c = -matrix(mu.values.astype(float))

        # Get constraints (if any)
        G_user, h_user, A_user, b_user = self.constraints.to_GhAb()

        # Inequality constraints: 0 <= x <= 0.2 (tighter max weight to reduce drawdown)
        G_ineq = np.vstack([np.eye(n), -np.eye(n)])
        h_ineq = np.hstack([np.full(n, 0.2), np.zeros(n)])

        if G_user is not None and h_user is not None:
            try:
                G_ineq = np.vstack([G_ineq, np.asarray(G_user)])
                h_ineq = np.hstack([h_ineq, np.asarray(h_user).flatten()])
            except:
                print("Warning: Could not combine inequality constraints")

        G = matrix(G_ineq.astype(float))
        h = matrix(h_ineq.astype(float))

        # Equality constraint: sum(x) = 1
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        if A_user is not None and b_user is not None:
            try:
                A_eq = np.vstack([A_eq, np.asarray(A_user)])
                b_eq = np.hstack([b_eq, np.asarray(b_user).flatten()])
            except:
                print("Warning: Could not combine equality constraints")

        A = matrix(A_eq.astype(float))
        b = matrix(b_eq.astype(float))

        solvers.options['show_progress'] = False
        try:
            sol = solvers.lp(c, G, h, A, b, solver='glpk')
            if sol['status'] != 'optimal':
                print("Primary solve failed, attempting relaxed constraints...")
                G_ineq = np.vstack([np.eye(n), -np.eye(n)])
                h_ineq = np.hstack([np.ones(n), np.zeros(n)])
                G = matrix(G_ineq.astype(float))
                h = matrix(h_ineq.astype(float))
                sol = solvers.lp(c, G, h, A, b, solver='glpk')

                if sol['status'] != 'optimal':
                    raise RuntimeError(f"Solver failed with status: {sol['status']}")

            weights = pd.Series(np.array(sol['x']).flatten(), index=mu.index)
            return {'weights': weights, 'status': 'optimal'}

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            equal_weights = pd.Series(1 / n, index=mu.index)
            return {'weights': equal_weights, 'status': 'failed', 'message': str(e)}