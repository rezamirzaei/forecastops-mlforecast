"""
MLForecast model factory.

This module provides factory functions to construct MLForecast instances
with pre-configured models, lag transforms, and date features suitable
for financial time-series forecasting.

Key components:
- ModelRegistry: Registry for model configurations
- default_models(): Returns a dictionary of sklearn-compatible regressors
- build_mlforecast(): Constructs an MLForecast instance with full configuration

Models included:
- Linear: Ridge, ElasticNet (regularized for stability)
- Tree Ensembles: RandomForest, ExtraTrees
- Gradient Boosting: HistGradientBoosting, LightGBM, XGBoost
- Neural Network: MLPRegressor (captures non-linear patterns)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExpandingStd,
    ExponentiallyWeightedMean,
    RollingMean,
    RollingStd,
    SeasonalRollingMean,
)
from mlforecast.target_transforms import Differences, LocalStandardScaler
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from mlforecast_realworld.config import ForecastSettings

# ============================================================================
# Model Configuration
# ============================================================================

class ModelComplexity(Enum):
    """Model complexity levels."""
    SIMPLE = "simple"      # Fast, interpretable (linear models)
    MODERATE = "moderate"  # Balanced (tree ensembles)
    COMPLEX = "complex"    # Powerful but slower (boosting, neural nets)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model."""
    name: str
    complexity: ModelComplexity
    factory: Callable[[int], Any]
    description: str


# ============================================================================
# Constants
# ============================================================================

# Model hyperparameters - optimized for financial returns prediction
DEFAULT_N_ESTIMATORS_RF = 150
DEFAULT_N_ESTIMATORS_ETR = 200
DEFAULT_N_ESTIMATORS_GB = 200
DEFAULT_N_ESTIMATORS_HGB = 250
DEFAULT_N_ESTIMATORS_LGBM = 200
DEFAULT_N_ESTIMATORS_XGB = 200
DEFAULT_LEARNING_RATE = 0.03  # Lower for better generalization
DEFAULT_LEARNING_RATE_NN = 0.001
DEFAULT_MAX_DEPTH_TREE = 10
DEFAULT_MAX_DEPTH_BOOST = 6
DEFAULT_MIN_SAMPLES_LEAF = 5
DEFAULT_MIN_SAMPLES_LEAF_HGB = 25

# Lag transform window sizes
ROLLING_WINDOW_5 = 5
ROLLING_WINDOW_10 = 10
ROLLING_WINDOW_21 = 21
ROLLING_WINDOW_63 = 63  # Quarterly
EWM_ALPHA = 0.2


# ============================================================================
# Model Factory Functions
# ============================================================================

def create_ridge(random_state: int) -> Ridge:
    """Create Ridge regression model (L2 regularization)."""
    return Ridge(alpha=1.0, random_state=random_state)


def create_elastic_net(random_state: int) -> ElasticNet:
    """Create ElasticNet model (L1 + L2 regularization)."""
    return ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=random_state,
    )


def create_sgd_regressor(random_state: int) -> SGDRegressor:
    """
    Create SGD Regressor using Stochastic Gradient Descent.

    SGD is faster to converge on large datasets and supports:
    - Online learning (can be updated incrementally)
    - Various loss functions and penalties
    - Efficient for high-dimensional data
    """
    return SGDRegressor(
        loss="huber",  # Robust to outliers (common in financial data)
        penalty="elasticnet",  # Combines L1 and L2 regularization
        alpha=0.0001,
        l1_ratio=0.15,
        max_iter=1000,
        tol=1e-4,
        learning_rate="adaptive",  # Adapts learning rate during training
        eta0=0.01,  # Initial learning rate
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )


def create_random_forest(random_state: int) -> RandomForestRegressor:
    """Create Random Forest regressor."""
    return RandomForestRegressor(
        n_estimators=DEFAULT_N_ESTIMATORS_RF,
        max_depth=DEFAULT_MAX_DEPTH_TREE,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def create_extra_trees(random_state: int) -> ExtraTreesRegressor:
    """Create Extra Trees regressor."""
    return ExtraTreesRegressor(
        n_estimators=DEFAULT_N_ESTIMATORS_ETR,
        max_depth=DEFAULT_MAX_DEPTH_TREE + 2,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def create_gradient_boosting(random_state: int) -> GradientBoostingRegressor:
    """Create Gradient Boosting regressor with stochastic settings for faster training."""
    return GradientBoostingRegressor(
        n_estimators=DEFAULT_N_ESTIMATORS_GB,
        learning_rate=DEFAULT_LEARNING_RATE,
        max_depth=DEFAULT_MAX_DEPTH_BOOST,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        subsample=0.7,  # Stochastic gradient boosting - use subset of samples
        max_features=0.8,  # Stochastic - use subset of features per tree
        random_state=random_state,
    )


def create_hist_gradient_boosting(random_state: int) -> HistGradientBoostingRegressor:
    """Create Histogram Gradient Boosting regressor."""
    return HistGradientBoostingRegressor(
        max_iter=DEFAULT_N_ESTIMATORS_HGB,
        learning_rate=DEFAULT_LEARNING_RATE,
        max_depth=DEFAULT_MAX_DEPTH_BOOST,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF_HGB,
        l2_regularization=0.1,
        random_state=random_state,
    )


def create_lightgbm(random_state: int) -> LGBMRegressor:
    """Create LightGBM regressor with stochastic settings for faster convergence."""
    return LGBMRegressor(
        n_estimators=DEFAULT_N_ESTIMATORS_LGBM,
        learning_rate=DEFAULT_LEARNING_RATE,
        max_depth=DEFAULT_MAX_DEPTH_BOOST,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.7,  # Stochastic - row sampling
        subsample_freq=1,  # Apply subsampling every iteration
        colsample_bytree=0.7,  # Stochastic - column sampling
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
        boosting_type="gbdt",  # Can also use "dart" for dropout
    )


def create_xgboost(random_state: int) -> XGBRegressor:
    """Create XGBoost regressor with stochastic settings for faster convergence."""
    return XGBRegressor(
        n_estimators=DEFAULT_N_ESTIMATORS_XGB,
        learning_rate=DEFAULT_LEARNING_RATE,
        max_depth=DEFAULT_MAX_DEPTH_BOOST,
        min_child_weight=5,
        subsample=0.7,  # Stochastic - row sampling per tree
        colsample_bytree=0.7,  # Stochastic - column sampling per tree
        colsample_bylevel=0.8,  # Column sampling per level
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",  # Faster histogram-based algorithm
    )


def create_mlp(random_state: int) -> MLPRegressor:
    """
    Create MLP (Neural Network) regressor.

    Architecture designed for financial time series:
    - Multiple hidden layers for non-linear pattern capture
    - Regularization to prevent overfitting
    - SGD optimizer for faster convergence on large datasets
    """
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="sgd",  # Stochastic Gradient Descent for faster convergence
        alpha=0.01,  # L2 regularization
        learning_rate="adaptive",
        learning_rate_init=DEFAULT_LEARNING_RATE_NN,
        momentum=0.9,  # SGD momentum for faster convergence
        nesterovs_momentum=True,  # Nesterov's accelerated gradient
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Registry for model configurations.

    Provides a centralized way to manage and create models
    based on complexity level or specific selection.
    """

    _models: dict[str, ModelConfig] = {
        "ridge": ModelConfig(
            name="ridge",
            complexity=ModelComplexity.SIMPLE,
            factory=create_ridge,
            description="Ridge regression with L2 regularization",
        ),
        "enet": ModelConfig(
            name="enet",
            complexity=ModelComplexity.SIMPLE,
            factory=create_elastic_net,
            description="ElasticNet with L1+L2 regularization",
        ),
        "sgd": ModelConfig(
            name="sgd",
            complexity=ModelComplexity.SIMPLE,
            factory=create_sgd_regressor,
            description="SGD Regressor - fast convergence with stochastic gradient descent",
        ),
        "rf": ModelConfig(
            name="rf",
            complexity=ModelComplexity.MODERATE,
            factory=create_random_forest,
            description="Random Forest ensemble",
        ),
        "etr": ModelConfig(
            name="etr",
            complexity=ModelComplexity.MODERATE,
            factory=create_extra_trees,
            description="Extra Trees ensemble",
        ),
        "gb": ModelConfig(
            name="gb",
            complexity=ModelComplexity.COMPLEX,
            factory=create_gradient_boosting,
            description="Gradient Boosting (sklearn)",
        ),
        "hgb": ModelConfig(
            name="hgb",
            complexity=ModelComplexity.COMPLEX,
            factory=create_hist_gradient_boosting,
            description="Histogram Gradient Boosting",
        ),
        "lgbm": ModelConfig(
            name="lgbm",
            complexity=ModelComplexity.COMPLEX,
            factory=create_lightgbm,
            description="LightGBM gradient boosting",
        ),
        "xgb": ModelConfig(
            name="xgb",
            complexity=ModelComplexity.COMPLEX,
            factory=create_xgboost,
            description="XGBoost gradient boosting",
        ),
        "mlp": ModelConfig(
            name="mlp",
            complexity=ModelComplexity.COMPLEX,
            factory=create_mlp,
            description="Multi-layer Perceptron neural network",
        ),
    }

    @classmethod
    def get_model(cls, name: str, random_state: int) -> Any:
        """Get a model instance by name."""
        config = cls._models.get(name)
        if config is None:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return config.factory(random_state)

    @classmethod
    def get_models_by_complexity(
        cls, complexity: ModelComplexity, random_state: int
    ) -> dict[str, Any]:
        """Get all models matching a complexity level."""
        return {
            name: config.factory(random_state)
            for name, config in cls._models.items()
            if config.complexity == complexity
        }

    @classmethod
    def get_all_models(cls, random_state: int) -> dict[str, Any]:
        """Get all registered models."""
        return {
            name: config.factory(random_state)
            for name, config in cls._models.items()
        }

    @classmethod
    def list_models(cls) -> list[dict[str, str]]:
        """List all available models with descriptions."""
        return [
            {
                "name": config.name,
                "complexity": config.complexity.value,
                "description": config.description,
            }
            for config in cls._models.values()
        ]

    @classmethod
    def register(cls, config: ModelConfig) -> None:
        """Register a new model configuration."""
        cls._models[config.name] = config


# ============================================================================
# Date Feature Functions
# ============================================================================

def week_of_month(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """
    Calculate the week of month (1-5) for given dates.

    Args:
        dates: DatetimeIndex or Series with datetime values.

    Returns:
        Series with week of month values (1 = first week, etc.).
    """
    if hasattr(dates, "dt"):
        day = dates.dt.day
    else:
        day = pd.Index(dates).day
    return ((day - 1) // 7) + 1


# ============================================================================
# Transform Naming
# ============================================================================

def lag_transform_namer(tfm: Any, lag: int, *args: Any) -> str:
    """
    Generate consistent feature names for lag transforms.

    Args:
        tfm: The transform class or instance.
        lag: The lag value being transformed.
        *args: Additional arguments passed to the transform.

    Returns:
        A descriptive feature name like 'rollingmean_lag7_7'.
    """
    base = tfm.__name__ if hasattr(tfm, "__name__") else tfm.__class__.__name__.lower()
    if args:
        args_part = "_".join(str(arg) for arg in args)
        return f"{base}_lag{lag}_{args_part}"
    return f"{base}_lag{lag}"


# ============================================================================
# Model Factory
# ============================================================================

def default_models(random_state: int) -> dict[str, Any]:
    """
    Create the default ensemble of regression models for forecasting.

    This ensemble covers a range of model families optimized for
    financial returns prediction:
    - Linear models: Ridge, ElasticNet (regularized for stability)
    - Tree ensembles: RandomForest, ExtraTrees (feature interactions)
    - Gradient boosting: GB, HGB, LightGBM, XGBoost (non-linear patterns)
    - Neural network: MLP (complex non-linear relationships)

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping model names to sklearn-compatible estimators.
    """
    return ModelRegistry.get_all_models(random_state)


def get_models_for_complexity(
    complexity: ModelComplexity, random_state: int
) -> dict[str, Any]:
    """
    Get models filtered by complexity level.

    Args:
        complexity: Desired model complexity level.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of models matching the complexity level.
    """
    return ModelRegistry.get_models_by_complexity(complexity, random_state)


def build_mlforecast(settings: ForecastSettings) -> MLForecast:
    """
    Build a fully configured MLForecast instance from settings.

    This factory configures:
    - Multiple regression models via ModelRegistry
    - Extended lag transforms for returns prediction
    - Date features: dayofweek, month, quarter, week_of_month
    - Target transforms: differencing and local standardization

    Args:
        settings: ForecastSettings with horizon, lags, frequency, etc.

    Returns:
        Configured MLForecast instance ready for fit/predict.
    """
    season_length = int(settings.season_length)

    # Extended lag transforms for financial returns
    lag_transforms = {
        1: [ExpandingMean(), ExpandingStd()],
        ROLLING_WINDOW_5: [
            RollingMean(window_size=ROLLING_WINDOW_5),
            RollingStd(window_size=ROLLING_WINDOW_5),
        ],
        ROLLING_WINDOW_10: [
            RollingMean(window_size=ROLLING_WINDOW_10),
            ExponentiallyWeightedMean(alpha=EWM_ALPHA),
        ],
        ROLLING_WINDOW_21: [
            RollingMean(window_size=ROLLING_WINDOW_21),
            RollingStd(window_size=ROLLING_WINDOW_21),
            SeasonalRollingMean(season_length=season_length, window_size=4),
        ],
    }

    date_features: list[Any] = ["dayofweek", "month", "quarter", week_of_month]
    target_transforms: list[Any] = [Differences([1]), LocalStandardScaler()]

    return MLForecast(
        models=default_models(settings.random_state),
        freq=settings.freq,
        lags=settings.lags,
        lag_transforms=lag_transforms,  # type: ignore[arg-type]
        date_features=date_features,
        num_threads=settings.num_threads,
        target_transforms=target_transforms,
        lag_transforms_namer=lag_transform_namer,
    )
