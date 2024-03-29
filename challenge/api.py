import fastapi
import pandas as pd
from typing import Dict, List
from challenge.classes import FlightsIn
from challenge.model import DelayModel

app = fastapi.FastAPI()

delay_model = DelayModel()

train = pd.read_csv(
    "data/data.csv",
    usecols=["Fecha-I", "Fecha-O", "MES", "TIPOVUELO", "OPERA"]
)

train, label = delay_model.preprocess(train, target_column="delay")
delay_model.fit(train, label)


@app.get("/", status_code=200)
async def root() -> Dict[str, str]:
    return {
        "message": "MLE Challenge - Facundo M. Scasso"
    }


@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(flights: FlightsIn) -> Dict[str, List[int]]:
    """Predicts if the `flights` inputed are going to be delayed or not.
    Note: the predictions are independent from each other.

    Args:
        flights (List[Dict[str, Any]]): Flights' features for the predictions

    Returns:
        dict: Predictions as a list of 1s and 0s
    """
    if "flights" not in flights:
        raise fastapi.exceptions.HTTPException(
            fastapi.status.HTTP_400_BAD_REQUEST,
            "Missing 'flights' key"
        )
    assert delay_model.model_is_set
    delay_model.validate_flights(flights["flights"])

    features = pd.DataFrame(flights["flights"])
    features = delay_model.preprocess(features)

    return {
        "predict": delay_model.predict(features)
    }
