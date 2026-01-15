from .model_factory_interface import ModelFactoryInterface
from .distributed_model import DistributedModel
from model.business_layer.forecasting.cluster_spec_selector import ClusterSpecSelector
from typing import Dict, Any, List

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
