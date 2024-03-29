from typing import List, Dict, Any, Callable, Literal

FeatureName = str
FeatureValue = Any

Flight = Dict[FeatureName, FeatureValue]
Flights = List[Flight]
FlightsIn = Dict[Literal["flights"], Flights]

Validator = Callable[[Any], bool]
ValidatorDict = Dict[FeatureName, Validator]

ModelType = Literal["xgb", "lr"]
