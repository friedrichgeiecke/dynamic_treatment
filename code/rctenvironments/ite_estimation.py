# Doubly robust individual treatment effect estimation function from Adusumilli, Geiecke, Schilter (2024)

import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression


def add_doubly_robust_ites(df, outcome, treatment_indicator, covariates):
    """

    Adds doubly robust individual treatment effect estimates to the data frame that can
    be used as rewards in an RL environment.

    Inputs

    df: Pandas data frame with outcome column, treatment indicator column, and covariate
    columns
    outcome: Name of outcome column (string)
    treatment_indicator: Name of treatment indicator column (string); treatment
    indicator is in {0,1}
    covariates: Column names of covariates (list of strings)

    Outputs

    Data frame with added 'ite' and 'ite_ols' columns

    """

    X0 = df.loc[df[treatment_indicator] == 0, covariates]
    y0 = df.loc[df[treatment_indicator] == 0, outcome]
    X1 = df.loc[df[treatment_indicator] == 1, covariates]
    y1 = df.loc[df[treatment_indicator] == 1, outcome]

    # 1. Predicting yhat of the other other group - no folds used

    def predict_other_group(X, y, Xother):
        full_lm = LinearRegression().fit(X, y)
        yhatother = pd.Series(full_lm.predict(Xother))
        yhatother.index = Xother.index
        return yhatother

    y0hat_group_1 = predict_other_group(X0, y0, X1)
    df.loc[y0hat_group_1.index, "y0hat"] = y0hat_group_1
    y1hat_group_0 = predict_other_group(X1, y1, X0)
    df.loc[y1hat_group_0.index, "y1hat"] = y1hat_group_0

    # 2. Predicting yhat of the own group (with leave one out folds)

    def predict_same_group(X, y):
        loocf = LeaveOneOut()
        yhat = []
        for train, test in loocf.split(X):
            loocf_lm = LinearRegression().fit(X.iloc[train, :], y.iloc[train])
            yhat.append(loocf_lm.predict(X.iloc[test, :])[0])
        yhat = pd.Series(yhat)
        yhat.index = X.index
        return yhat

    y0hat_group_0 = predict_same_group(X0, y0)
    df.loc[y0hat_group_0.index, "y0hat"] = y0hat_group_0
    y1hat_group_1 = predict_same_group(X1, y1)
    df.loc[y1hat_group_1.index, "y1hat"] = y1hat_group_1

    # 3. Computing the final ITEs

    # Propensity score (constant because RCT)
    ps = df[treatment_indicator].sum() / df.shape[0]

    # Compute ITEs
    for i in df.index:
        w = df.loc[i, treatment_indicator]
        y1hat = df.loc[i, "y1hat"]
        y0hat = df.loc[i, "y0hat"]
        y = df.loc[i, outcome]

        # ITE of observation
        df.loc[i, "ite"] = (
            y1hat
            - y0hat
            + (2 * w - 1)
            * ((y - (1 - w) * y0hat - w * y1hat) / (w * ps + (1 - w) * (1 - ps)))
        )

    # OLS-only based ITE
    df["ite_ols"] = df["y1hat"] - df["y0hat"]

    # Return df including new columns
    return df
