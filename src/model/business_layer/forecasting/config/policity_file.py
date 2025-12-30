from .model_spec import ModelSpec
from typing import Dict


MODEL_SPECS_BY_CLASS: Dict[str, ModelSpec] = {
    'Smooth': ModelSpec(
        name='smooth_lgbm',
        models = {
            'lgbm':{'type': 'lgbm', 'params':{'learning_rate':0.01, 'n_estimators':1000}}
        }
    ),

    'Erratic': ModelSpec(
        name = 'erratic_xgb',
        models = {
            'xgb':{'type': 'xgb', 'params':{'n_estimators':500, 'max_depth':6}}
        }
    ),
    'Lumpy': ModelSpec(
        name = 'lumpy_lgbm',
        models = {
            'lgbm': {'type':'lgbm', 'params':{'n_estimators':200, 'num_leaves':21, 'max_depth':10}}
        }
    ),
    'Intermittent': ModelSpec(
        name = 'intermittent_xgb',
        models = {
            'xgb': {'type': 'xgb', 'params':{'n_estimators':300, 'learning_rate':0.05, 'subsample':0.8}}
        }
    )

}