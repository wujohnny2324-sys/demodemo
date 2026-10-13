"""Nelson-Siegel yield curve factor estimation, prediction, and KRD-based return calculation."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize
from fds.fpe.quant.utils import stepwise_regression_selection


def generate_exposures(t, decay):
    """
    Generate the Nelson-Siegel factor exposure matrix.

    Returns an (N x 3) array [1, loading1, loading2] for each tenor in t.
    """
    t = np.asarray(t, dtype=float)
    factor0 = np.ones(len(t))
    factor1 = decay * (1 - np.exp(-t / decay)) / t
    factor2 = decay * (1 - np.exp(-t / decay)) / t - np.exp(-t / decay)
    return np.array([factor0, factor1, factor2]).transpose()


def _fit_error(decay, t, y):
    """Sum of squared OLS residuals at the given decay parameter (used for optimization)."""
    factors = generate_exposures(t, decay)
    model = sm.OLS(y, factors).fit()
    return np.sum((y - model.fittedvalues) ** 2)


def get_betas(t, y, decay):
    """Estimate Nelson-Siegel factor loadings [beta0, beta1, beta2] via OLS."""
    factors = generate_exposures(t, decay)
    return sm.OLS(y, factors).fit().params


def find_decay_for_tenor(target_tenor_months, x0=15.0):
    """
    Find the decay parameter that maximises curvature-factor exposure at a target tenor.

    This is the standard way to fix lambda: choose the tenor where you expect the
    most curvature variation and solve analytically/numerically for the optimum.

    Args:
        target_tenor_months: Tenor in months at which curvature exposure is maximised.
        x0: Initial optimiser guess.

    Returns:
        Optimal decay parameter (float).
    """
    def neg_curvature(decay, t):
        return -1.0 * (decay * (1 - np.exp(-t / decay)) / t - np.exp(-t / decay))

    return optimize.minimize(neg_curvature, x0=x0, args=(target_tenor_months,)).x[0]


def estimate_ns_factors(yield_data, decay=None, symbol='TRYUS-FDS',
                        decay_bounds=(25.0, 35.0), x0=15.0):
    """
    Estimate Nelson-Siegel [beta0, beta1, beta2, decay] over time.

    Args:
        yield_data: DataFrame with (date, symbol) MultiIndex and columns 'tenor', 'yield'.
        decay: Fixed decay parameter. Pass None to optimise per date (variable decay).
        symbol: Yield-curve identifier used as the second index level.
        decay_bounds: (min, max) to clamp the optimised decay when decay=None.
        x0: Initial guess for per-date decay optimisation.

    Returns:
        DataFrame indexed by date with columns [beta0, beta1, beta2, decay].
    """
    params = {}
    dates = yield_data.index.get_level_values('date').unique()

    for i, date in enumerate(dates):
        t = yield_data.loc[date, symbol]['tenor'].to_numpy()
        y = yield_data.loc[date, symbol]['yield'].to_numpy()

        if decay is None:
            try:
                decay_t = optimize.minimize(fun=_fit_error, x0=x0, args=(t, y)).x[0]
                decay_t = float(np.clip(decay_t, decay_bounds[0], decay_bounds[1]))
            except Exception:
                if i > 0:
                    params[date] = params[dates[i - 1]]
                    continue
                else:
                    decay_t = x0
        else:
            decay_t = float(decay)

        beta = get_betas(t, y, decay_t)

        # Reject implausibly large slope values and reuse previous period
        if np.abs(beta[1]) > 20 and i > 0:
            params[date] = params[dates[i - 1]]
            continue

        params[date] = {
            'beta0': float(beta[0]),
            'beta1': float(beta[1]),
            'beta2': float(beta[2]),
            'decay': decay_t,
        }

    return pd.DataFrame(params).transpose().astype(float)


def add_ns_deltas(ns_factors):
    """
    Add forward and trailing one-period change columns to a Nelson-Siegel DataFrame.

    Adds six columns: beta{0,1,2}_delta_fwd and beta{0,1,2}_delta.

    Args:
        ns_factors: DataFrame with columns [beta0, beta1, beta2, decay].

    Returns:
        Copy of the DataFrame with additional delta columns.
    """
    df = ns_factors.copy()
    for b in ['beta0', 'beta1', 'beta2']:
        df[f'{b}_delta_fwd'] = df[b].shift(-1) - df[b]
        df[f'{b}_delta']     = df[b] - df[b].shift(1)
    return df


def predict_ns_factors(regression_data, X_columns):
    """
    Use stepwise OLS regression to predict forward changes in each Nelson-Siegel factor.

    Runs one stepwise selection + OLS fit per factor (beta0, beta1, beta2).

    Args:
        regression_data: DataFrame containing X_columns and beta{0,1,2}_delta_fwd columns.
        X_columns: Candidate regressor column names.

    Returns:
        Dict keyed by 'beta0', 'beta1', 'beta2'. Each value is a dict with:
            'model'               : fitted statsmodels OLS result
            'selected_regressors' : list of column names chosen by stepwise selection
    """
    results = {}
    for beta_name in ['beta0', 'beta1', 'beta2']:
        target = f'{beta_name}_delta_fwd'
        stepwise = stepwise_regression_selection(
            X=regression_data[X_columns],
            y=regression_data[target].squeeze(),
        )
        selected = stepwise['selected_signals']
        X_sel = sm.add_constant(regression_data[selected])
        model = sm.OLS(regression_data[target], X_sel, missing='drop').fit()
        results[beta_name] = {'model': model, 'selected_regressors': selected}
    return results


def get_yc_at_tenors(date, ns_factors, tenors_months):
    """
    Evaluate the fitted Nelson-Siegel yield curve at specified tenors on a given date.

    Args:
        date: Date key into ns_factors index.
        ns_factors: DataFrame with [beta0, beta1, beta2, decay] indexed by date.
        tenors_months: Array-like of tenors in months.

    Returns:
        numpy array of fitted yield values.
    """
    decay_t = float(ns_factors.loc[date, 'decay'])
    betas   = ns_factors.loc[date, ['beta0', 'beta1', 'beta2']].to_numpy()
    factors = generate_exposures(np.array(tenors_months, dtype=float), decay_t)
    return np.matmul(factors, betas)


def get_predicted_yc_at_tenors(date, ns_factors, fitted_beta_changes, tenors_months):
    """
    Evaluate the predicted next-period yield curve at specified tenors.

    Adds model-fitted forward beta deltas to the current N/S factors, then evaluates.

    Args:
        date: Date key into ns_factors and fitted_beta_changes index.
        ns_factors: DataFrame with [beta0, beta1, beta2, decay] indexed by date.
        fitted_beta_changes: DataFrame with columns ['Fitted b0','Fitted b1','Fitted b2']
                             (model-fitted forward deltas from predict_ns_factors).
        tenors_months: Array-like of tenors in months.

    Returns:
        numpy array of predicted yield values.
    """
    decay_t       = float(ns_factors.loc[date, 'decay'])
    betas         = ns_factors.loc[date, ['beta0', 'beta1', 'beta2']].to_numpy()
    fwd_deltas    = fitted_beta_changes.loc[date, ['Fitted b0', 'Fitted b1', 'Fitted b2']].to_numpy()
    predicted     = betas + fwd_deltas
    factors       = generate_exposures(np.array(tenors_months, dtype=float), decay_t)
    return np.matmul(factors, predicted)


def calc_krd_return(krd_array, yield_change):
    """
    Calculate the price-return approximation from a parallel yield shift using KRDs.

    Return ≈ -KRD · Δy  (first-order duration approximation).

    Args:
        krd_array: Array of KRD values at each tenor key rate.
        yield_change: Array of yield changes (in percentage points) at each KRD tenor.

    Returns:
        Scalar predicted return (float).
    """
    return float(-np.dot(krd_array, yield_change))
