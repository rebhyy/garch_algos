"""
GARCH Models Module

Implements GARCH family volatility models with comprehensive error handling,
validation, and diagnostics.

Models Supported:
- GARCH(1,1): Standard GARCH model
- EGARCH: Exponential GARCH (captures asymmetric volatility)
- GJR-GARCH: Threshold GARCH (models leverage effects)
- APARCH: Asymmetric Power ARCH (most flexible)

Author: GARCH Algo Intelligence Platform
License: MIT
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GarchFit:
    """
    Container for GARCH model fit results.

    Attributes:
        name: Model name (e.g., 'GARCH11', 'EGARCH')
        result: ARCHModelResult object from arch package
        conditional_volatility: Time series of conditional volatility σ(t)
        one_step_forecast: One-step ahead volatility forecast
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        converged: Whether model fitting converged
        num_obs: Number of observations used
    """
    name: str
    result: ARCHModelResult
    conditional_volatility: pd.Series
    one_step_forecast: float
    aic: float
    bic: float
    converged: bool
    num_obs: int

    def __repr__(self) -> str:
        return (
            f"GarchFit(name='{self.name}', "
            f"AIC={self.aic:.2f}, BIC={self.bic:.2f}, "
            f"σ(1)={self.one_step_forecast:.4f}, converged={self.converged})"
        )


class GARCHModelError(Exception):
    """Base exception for GARCH modeling errors."""
    pass


class InsufficientDataError(GARCHModelError):
    """Raised when insufficient data is provided for GARCH fitting."""
    pass


class ConvergenceError(GARCHModelError):
    """Raised when GARCH model fails to converge."""
    pass


def validate_returns(returns: pd.Series, min_obs: int = 100) -> pd.Series:
    """
    Validate returns data for GARCH modeling.

    Args:
        returns: Log returns time series
        min_obs: Minimum number of observations required

    Returns:
        Cleaned returns series

    Raises:
        InsufficientDataError: If data is insufficient or invalid
    """
    if not isinstance(returns, pd.Series):
        raise InsufficientDataError("Returns must be a pandas Series")

    # Remove NaN values
    clean_returns = returns.dropna()

    # Check for sufficient data
    if len(clean_returns) < min_obs:
        raise InsufficientDataError(
            f"Insufficient data: {len(clean_returns)} observations, "
            f"minimum {min_obs} required"
        )

    # Check for constant series
    if clean_returns.std() == 0:
        raise InsufficientDataError("Returns series is constant (zero variance)")

    # Check for extreme values (potential data errors)
    abs_max = clean_returns.abs().max()
    if abs_max > 1.0:  # 100% return in one period is suspicious for log returns
        logger.warning(
            f"Extreme value detected: {abs_max:.4f}. "
            "This may indicate data quality issues."
        )

    # Check for infinite values
    if np.isinf(clean_returns).any():
        raise InsufficientDataError("Returns contain infinite values")

    return clean_returns


def _fit_single_model(
    name: str,
    model_instance,
    index: pd.DatetimeIndex,
    require_convergence: bool = True
) -> GarchFit:
    """
    Fit a single GARCH model with error handling.

    Args:
        name: Model identifier
        model_instance: arch_model instance
        index: DatetimeIndex for the series
        require_convergence: If True, raise error on non-convergence

    Returns:
        GarchFit object with results

    Raises:
        ConvergenceError: If model doesn't converge and required
        GARCHModelError: For other fitting errors
    """
    try:
        # Fit model with suppressed output
        result = model_instance.fit(disp="off", show_warning=False)

        # Extract conditional volatility
        cond_vol = pd.Series(
            result.conditional_volatility,
            index=index,
            name=f"{name}_sigma"
        )

        # Get one-step ahead forecast
        forecast = result.forecast(horizon=1)
        sigma_forecast = float(np.sqrt(forecast.variance.iloc[-1, 0]))

        # Check convergence
        converged = result.convergence_flag == 0
        if not converged and require_convergence:
            raise ConvergenceError(
                f"{name} model failed to converge. "
                f"Convergence flag: {result.convergence_flag}"
            )

        if not converged:
            logger.warning(f"{name} did not converge properly")

        return GarchFit(
            name=name,
            result=result,
            conditional_volatility=cond_vol,
            one_step_forecast=sigma_forecast,
            aic=result.aic,
            bic=result.bic,
            converged=converged,
            num_obs=result.nobs
        )

    except ConvergenceError:
        raise
    except Exception as e:
        raise GARCHModelError(
            f"Error fitting {name} model: {str(e)}"
        ) from e


def fit_garch_family(
    returns: pd.Series,
    scale: float = 100.0,
    min_obs: int = 100,
    require_convergence: bool = False
) -> Dict[str, GarchFit]:
    """
    Fit all GARCH family models to returns data.

    Fits four models:
    - GARCH(1,1): Standard GARCH with Student's t distribution
    - EGARCH(1,1): Exponential GARCH for asymmetric effects
    - GJR-GARCH(1,1,1): Threshold GARCH with power=2.0
    - APARCH(1,1): Asymmetric Power ARCH

    Args:
        returns: Log returns series (decimal form, e.g., 0.01 for 1%)
        scale: Scaling factor for returns (default 100 for percentage)
        min_obs: Minimum observations required
        require_convergence: If True, raise error if any model doesn't converge

    Returns:
        Dictionary mapping model names to GarchFit objects

    Raises:
        InsufficientDataError: If data validation fails
        ConvergenceError: If require_convergence=True and model doesn't converge
        GARCHModelError: For other fitting errors

    Example:
        >>> returns = np.log(prices).diff().dropna()
        >>> fits = fit_garch_family(returns)
        >>> best = best_by_aic(fits)
        >>> print(f"Best model: {best.name}, AIC: {best.aic:.2f}")
    """
    logger.info(f"Fitting GARCH family models on {len(returns)} observations")

    # Validate input data
    clean_returns = validate_returns(returns, min_obs=min_obs)

    # Scale returns (arch expects percentage returns)
    y = clean_returns * scale

    fits: Dict[str, GarchFit] = {}

    # Define models to fit
    models = {
        "GARCH11": arch_model(
            y, vol="GARCH", p=1, q=1, dist="t", mean="constant"
        ),
        "EGARCH": arch_model(
            y, vol="EGARCH", p=1, q=1, dist="t", mean="constant"
        ),
        "GJR": arch_model(
            y, vol="GARCH", p=1, o=1, q=1, power=2.0, dist="t", mean="constant"
        ),
        "APARCH": arch_model(
            y, vol="APARCH", p=1, q=1, dist="t", mean="constant"
        ),
    }

    # Fit each model
    failed_models = []
    for name, model in models.items():
        try:
            fits[name] = _fit_single_model(
                name, model, y.index, require_convergence
            )
            logger.info(f"✓ {name}: AIC={fits[name].aic:.2f}, BIC={fits[name].bic:.2f}")
        except (ConvergenceError, GARCHModelError) as e:
            logger.error(f"✗ {name}: {str(e)}")
            failed_models.append(name)
            if require_convergence:
                raise

    if not fits:
        raise GARCHModelError(
            "All GARCH models failed to fit. "
            f"Failed models: {', '.join(failed_models)}"
        )

    logger.info(f"Successfully fitted {len(fits)}/{len(models)} models")
    return fits


def best_by_aic(fits: Dict[str, GarchFit]) -> GarchFit:
    """
    Select best model by Akaike Information Criterion (AIC).

    Lower AIC indicates better model fit with parsimony penalty.

    Args:
        fits: Dictionary of GarchFit objects

    Returns:
        GarchFit object with lowest AIC

    Raises:
        ValueError: If fits dictionary is empty
    """
    if not fits:
        raise ValueError("No fitted models provided")

    best = min(fits.values(), key=lambda f: f.aic)
    logger.info(f"Best model by AIC: {best.name} (AIC={best.aic:.2f})")
    return best


def best_by_bic(fits: Dict[str, GarchFit]) -> GarchFit:
    """
    Select best model by Bayesian Information Criterion (BIC).

    Lower BIC indicates better model fit with stronger parsimony penalty than AIC.

    Args:
        fits: Dictionary of GarchFit objects

    Returns:
        GarchFit object with lowest BIC

    Raises:
        ValueError: If fits dictionary is empty
    """
    if not fits:
        raise ValueError("No fitted models provided")

    best = min(fits.values(), key=lambda f: f.bic)
    logger.info(f"Best model by BIC: {best.name} (BIC={best.bic:.2f})")
    return best


def annualize_volatility(
    sigma: float,
    bars_per_year: float
) -> float:
    """
    Annualize volatility from model frequency.

    Args:
        sigma: Volatility in model frequency (e.g., daily, hourly)
        bars_per_year: Number of bars per year

    Returns:
        Annualized volatility (%)

    Example:
        >>> # 30-minute bars: 252 days * 24 hours * 2 bars/hour
        >>> sigma_annual = annualize_volatility(sigma_30m, 252 * 24 * 2)
    """
    return sigma * np.sqrt(bars_per_year)


def get_model_diagnostics(fit: GarchFit) -> Dict[str, any]:
    """
    Extract diagnostic statistics from fitted GARCH model.

    Args:
        fit: GarchFit object

    Returns:
        Dictionary containing diagnostic statistics
    """
    result = fit.result

    diagnostics = {
        "model_name": fit.name,
        "aic": fit.aic,
        "bic": fit.bic,
        "log_likelihood": result.loglikelihood,
        "num_obs": fit.num_obs,
        "converged": fit.converged,
        "mean_volatility": fit.conditional_volatility.mean(),
        "std_volatility": fit.conditional_volatility.std(),
        "min_volatility": fit.conditional_volatility.min(),
        "max_volatility": fit.conditional_volatility.max(),
    }

    # Add standardized residuals statistics if available
    try:
        std_resid = result.std_resid
        diagnostics.update({
            "resid_mean": std_resid.mean(),
            "resid_std": std_resid.std(),
            "resid_skew": pd.Series(std_resid).skew(),
            "resid_kurt": pd.Series(std_resid).kurtosis(),
        })
    except Exception as e:
        logger.warning(f"Could not extract residual diagnostics: {e}")

    return diagnostics
