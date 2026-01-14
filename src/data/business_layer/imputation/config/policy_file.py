from business_layer.imputation.imputers import ImputerInterface, RollingMeanFFillImputer, RollingMedianFFillImputer, ZeroFillImputer
from typing import Dict

IMPUTER_BY_CLASS: Dict[str, ImputerInterface] = {
    'Smooth': RollingMeanFFillImputer(window_size = 3),
    'Erratic': RollingMedianFFillImputer(window_size = 4),
    'Intermittent': ZeroFillImputer(),
    'Lumpy':RollingMedianFFillImputer(window_size = 12)
}