from .model_factory_interface import ModelFactoryInterface
from .distributed_model import DistributedModel
from .local_model import LocalModel
from model.business_layer.forecasting.cluster_spec_selector import ClusterSpecSelector
from typing import Dict, Any, List, Tuple

class ModelFactory(ModelFactoryInterface):

    def built_models(self, models_config: Dict[str, Dict[str,Any]])->List[DistributedModel]:

        
        models: List[DistributedModel] = []
        
        for _, config in models_config.items():

            model_type: str = config.get("type")

            parameters: Dict[str,Any] = config.get('params', {})
            
            match model_type:

                case "lgbm":
                    from mlforecast.distributed.models.spark.lgb import SparkLGBMForecast
                    models.append(SparkLGBMForecast(**parameters))

                case "xgb":
                    from mlforecast.distributed.models.spark.xgb import SparkXGBForecast
                    models.append(SparkXGBForecast(**parameters))
                    
                case _:
                    raise ValueError(
                        f"Unsupported model_type='{model_type}'. Available: ['lgbm', 'xgb']."
                    )
                
        return  models
    
    def built_local_model(self, distributed_model: Dict[str,DistributedModel])->Tuple[str,LocalModel]:

        if len(distributed_model) != 1:

            raise ValueError(f"Expected exactly 1 model, got {len(distributed_model)}: {list(distributed_model.keys())}")
        
        model_name: str = next(iter(distributed_model.keys()))

        match model_name:

            case 'SparkXGBForecast':

                from mlforecast.distributed.models.spark.xgb import SparkXGBForecast

                xgboost: LocalModel = SparkXGBForecast().extract_local_model(trained_model = distributed_model[model_name])

                return 'XGBoostRegressor',xgboost
            
            
            case 'SparkLGBMForecast':

                lgbm: LocalModel = distributed_model[model_name]

                return 'LGBMRegressor', lgbm
            
        
            case _:
                raise ValueError(f"Unsupported model key: {model_name}")
        
