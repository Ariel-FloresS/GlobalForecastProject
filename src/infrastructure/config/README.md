# YAML configuration loader

Example usage to load the forecasting and imputation policies from the YAML file:

```python
from infrastructure.config import YamlConfigLoader

loader = YamlConfigLoader("src/infrastructure/config/policies/forecast_imputation_policies.yml")
config = loader.load()

forecasting_policies = loader.get_section("forecasting_policies")
imputation_policies = loader.get_section("imputation_policies")
```

Example to build objects from specs:

```python
from infrastructure.config import YamlConfigLoader

loader = YamlConfigLoader("src/infrastructure/config/policies/forecast_imputation_policies.yml")
loader.load()

imputers = {
    name: loader.build_object(spec)
    for name, spec in loader.get_section("imputation_policies").items()
}
```
