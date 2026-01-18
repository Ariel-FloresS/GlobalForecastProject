from pathlib import Path
from typing import Dict

from data.business_layer.imputation.imputers import (
    ImputerInterface,
    RollingMeanFFillImputer,
    RollingMedianFFillImputer,
    ZeroFillImputer,
)
from infrastructure.config import YamlConfigLoader

_POLICY_PATH = (
    Path(__file__).resolve().parents[4]
    / "infrastructure"
    / "config"
    / "policies"
    / "forecast_imputation_policies.yml"
)

_IMPUTER_REGISTRY = {
    "RollingMeanFFillImputer": RollingMeanFFillImputer,
    "RollingMedianFFillImputer": RollingMedianFFillImputer,
    "ZeroFillImputer": ZeroFillImputer,
}


def _load_imputers() -> Dict[str, ImputerInterface]:
    loader = YamlConfigLoader(_POLICY_PATH)
    loader.load()
    policies = loader.get_section("imputation_policies")
    return {
        policy_name: loader.build_object(policy_payload, registry=_IMPUTER_REGISTRY)
        for policy_name, policy_payload in policies.items()
    }


IMPUTER_BY_CLASS: Dict[str, ImputerInterface] = _load_imputers()
