import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Tuple, Union, List, Dict, Any
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from fastapi import status
from fastapi.exceptions import HTTPException
from challenge.classes import Flights, FeatureName, FeatureValue, ModelType
from challenge.validate import get_validators
from challenge.feat_eng import get_min_diff

VALIDATORS, FEAT_DTYPES = get_validators()
DELAY_THRESH_IN_MINS = 15
FEATURE_COLS = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]


class DelayModel:
    def __init__(
        self
    ):
        self._model = None
        self.model_type: Optional[ModelType] = None
        self.model_is_fitted = False
        self.input_cols: Optional[List[FeatureName]] = None
        self.dummies: Optional[Dict[FeatureName, List[FeatureValue]]] = None
        self.input_params: Optional[Dict[str, Any]] = None

        # self.set_model("lr")  # LogisticRegression
        self.set_model(
            "xgb",
            params={"random_state": 1, "learning_rate": 0.01}
        )  # XGBoost

    @property
    def model_is_set(self) -> bool:
        return self.model_type is not None

    @property
    def features(self) -> np.ndarray:
        assert self.model_is_set
        return self._model.feature_names_in_

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model_is_fitted:
            return data[self.features]
        else:
            with open("challenge/features_selected.txt") as f:
                features = f.readlines()
            features = [f[:-1] if f.endswith("\n") else f for f in features]
            return data[features]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        if target_column is None:
            if self.model_is_fitted:
                data = data[self.input_cols]
        else:
            data = self.process_delay(data)
            assert target_column in data.columns
            self.input_cols = [c for c in data.columns if c != target_column]

        data = data.astype(
            {
                k: v for k, v in FEAT_DTYPES.items()
                if k in data.columns and v != "DATETIME"
            }
        )

        # Getting dummies
        str_cols = {
            k: v for k, v in FEAT_DTYPES.items()
            if k in data.columns and v == "str"
        }
        if str_cols:
            if self.dummies is None:
                assert not self.model_is_fitted
                data = pd.concat(
                    [data.drop(str_cols, axis=1)] + [
                        pd.get_dummies(data[ft].astype(str), prefix=ft)
                        for ft in str_cols
                    ],
                    axis=1
                )
                self.dummies = {
                    ft: [
                        c[len(ft)+1:]
                        for c in data.columns if c.startswith(f"{ft}_")
                    ]
                    for ft in str_cols
                }
            else:
                assert self.model_is_fitted
                assert self.dummies is not None
                assert self.dummies
                cond = all(
                    v in self.dummies[ft]
                    for ft in str_cols
                    for v in data[ft].unique()
                )
                if not cond:
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        "Values used must be contained "
                        + "inside the model's fitted dummy columns"
                    )

                data = pd.concat(
                    [data.drop(str_cols, axis=1)] + [
                        pd.get_dummies(data[ft].astype(str), prefix=ft)
                        for ft in str_cols
                    ],
                    axis=1
                )
                missing_cols = [
                    f"{ft}_{v}"
                    for ft, vs in self.dummies.items()
                    for v in vs
                    if f"{ft}_{v}" not in data.columns
                ]
                if missing_cols:
                    data = pd.concat(
                        [data] + [
                            pd.DataFrame(
                                np.zeros(
                                    (data.shape[0], len(missing_cols)),
                                    dtype=int
                                ),
                                columns=missing_cols
                            )
                        ],
                        axis=1
                    )

        # data["period_day"] = data['Fecha-I'].apply(get_period_day)
        # data["high_season"] = data['Fecha-I'].apply(is_high_season)

        if target_column is not None:
            labels = pd.DataFrame(data[target_column])
            data = data.drop(target_column, axis=1)
            data = self._select_features(data)
            return data, labels
        else:
            data = self._select_features(data)
            return data

    def set_model(
        self,
        model_type: ModelType,
        params: Optional[dict] = None
    ) -> None:
        assert not self.model_is_set
        if params is not None:
            self.input_params = deepcopy(params)

        if model_type in {"xgb", "lr"}:
            self.model_type = model_type
        else:
            raise ValueError("model_type has to be of type 'xgb' or 'lr'.")

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        assert self.model_is_set and not self.model_is_fitted

        n1 = (target.values == 1).sum()
        n0 = target.shape[0] - n1
        _params = self.input_params if self.input_params is not None else {}

        if self.model_type == "lr":
            _params["class_weight"] = {
                1: n0 / target.size,
                0: n1 / target.size
            }
            self._model = LogisticRegression(**_params)
        elif self.model_type == "xgb":
            _params["scale_pos_weight"] = n0 / n1
            self._model = XGBClassifier(**_params)
        else:
            raise NotImplementedError

        self._model.fit(features, target)
        self.model_is_fitted = True

    def validate_flights(self, flights: Flights) -> None:
        assert self.model_is_set
        if not flights:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "There must be at least 1 flight to predict its delay status"
            )

        vars_set = set(self.input_cols)
        if not all(set(fl.keys()) == vars_set for fl in flights):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Wrong feature names"
                # if the API was private we could have more verbosity here
            )

        cond = all(
            VALIDATORS[feat](fl[feat])
            for feat in self.input_cols
            for fl in flights
        )
        if not cond:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Wrong feature values"
                # if the API was private we could have more verbosity here
            )

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        assert self.model_is_fitted
        return self._model.predict(features).astype(int).tolist()

    def unset_model(self) -> None:
        self.model_is_fitted = False
        self.dummies = None
        self.input_cols = None
        self.input_params = None
        self.model_type = None
        self._model = None
        print("Model has been unset")

    @staticmethod
    def process_delay(data: pd.DataFrame) -> pd.DataFrame:
        min_diff = data.apply(get_min_diff, axis=1)
        data = data.drop(["Fecha-I", "Fecha-O"], axis=1)
        data["delay"] = (min_diff > DELAY_THRESH_IN_MINS).astype(int)
        return data
