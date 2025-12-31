from .model_spec import ModelSpec
from mlforecast.lag_transforms import *
from mlforecast.target_transforms import *
from typing import Dict


MODEL_SPECS_BY_CLASS: Dict[str, ModelSpec] = {
    'Smooth': ModelSpec(
        name='smooth_lgbm_xgb',
        lags = [1,4],
        lag_transforms = {1:[RollingMean(window_size = 3)],
                        4:[RollingMean(window_size = 6)]},
        target_transforms = [Differences(1)],
        models = {
            'lgbm':{'type': 'lgbm', 'params':{'learning_rate':0.01, 'n_estimators':1000}},
            'xgb':{'type': 'xgb', 'params':{'n_estimators':500, 'max_depth':6}}
        }
    ),

    'Erratic': ModelSpec(
        name = 'erratic_xgb',
        lags=[1, 2, 3, 6, 12, 18, 24, 36],
        lag_transforms={
            1: [ExpandingStd()],
            3: [RollingMean(window_size=3, min_samples=1)],
            12: [ExponentiallyWeightedMean(alpha=0.5)],
        },
        target_transforms=[LocalStandardScaler()],
        models = {
            'xgb':{'type': 'xgb', 'params':{'n_estimators':500, 'max_depth':6}}
        }
    ),
    'Lumpy': ModelSpec(
        name = 'lumpy_lgbm',
        lags = [1, 2, 3, 6, 12, 18, 24],
        lag_transforms={
            1: [ExpandingStd()],
            6: [RollingMean(window_size=6, min_samples=1), RollingMean(window_size=12, min_samples=1)],
            12: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        target_transforms=[Differences([1]),LocalStandardScaler()],
        models = {
            'lgbm': {'type':'lgbm', 'params':{'n_estimators':200, 'num_leaves':21, 'max_depth':10}}
        }
    ),
    'Intermittent': ModelSpec(
        name = 'intermittent_xgb',
        lags=[1, 3, 6, 12],
        lag_transforms={
            1: [ExpandingStd()],
            12: [RollingMean(window_size=12, min_samples=1)],
        },
        target_transforms=[Differences([1])],
        models = {
            'xgb': {'type': 'xgb', 'params':{'n_estimators':300, 'learning_rate':0.05, 'subsample':0.8}}
        }
    )

}