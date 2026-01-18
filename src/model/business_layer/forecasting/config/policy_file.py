from pathlib import Path
from typing import Any, Dict

from mlforecast.lag_transforms import (
    ExpandingStd,
    ExponentiallyWeightedMean,
    RollingMean,
)
from mlforecast.target_transforms import Differences, LocalStandardScaler

from infrastructure.config import YamlConfigLoader
from .model_spec import ModelSpec

_POLICY_PATH = (
    Path(__file__).resolve().parents[4]
    / "infrastructure"
    / "config"
    / "policies"
    / "forecast_imputation_policies.yml"
)

_TRANSFORM_REGISTRY = {
    "RollingMean": RollingMean,
    "ExpandingStd": ExpandingStd,
    "ExponentiallyWeightedMean": ExponentiallyWeightedMean,
    "Differences": Differences,
    "LocalStandardScaler": LocalStandardScaler,
}


def _build_lag_transforms(
    loader: YamlConfigLoader,
    spec: dict[str, Any] | None,
) -> dict[int, list[Any]] | None:
    if not spec:
        return None
    transforms: dict[int, list[Any]] = {}
    for lag_key, transform_specs in spec.items():
        lag = int(lag_key)
        transforms[lag] = loader.build_object_list(transform_specs, registry=_TRANSFORM_REGISTRY)
    return transforms


def _build_target_transforms(
    loader: YamlConfigLoader,
    spec: list[Any] | None,
) -> list[Any] | None:
    if not spec:
        return None
    return loader.build_object_list(spec, registry=_TRANSFORM_REGISTRY)


def _build_model_spec(loader: YamlConfigLoader, payload: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=payload["name"],
        lags=payload.get("lags"),
        lag_transforms=_build_lag_transforms(loader, payload.get("lag_transforms")),
        target_transforms=_build_target_transforms(loader, payload.get("target_transforms")),
        models=payload.get("models", {}),
    )


def _load_model_specs() -> Dict[str, ModelSpec]:
    loader = YamlConfigLoader(_POLICY_PATH)
    loader.load()
    policies = loader.get_section("forecasting_policies")
    return {
        policy_name: _build_model_spec(loader, policy_payload["model_spec"])
        for policy_name, policy_payload in policies.items()
    }


MODEL_SPECS_BY_CLASS: Dict[str, ModelSpec] = _load_model_specs()
