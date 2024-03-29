import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List, Literal, Dict
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from fastapi import status
from fastapi.exceptions import HTTPException
from challenge.classes import Flights, FeatureName, FeatureValue
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
        self.model_is_fitted = False
        self.input_cols: Optional[List[FeatureName]] = None
        self.dummies: Optional[Dict[FeatureName, List[FeatureValue]]] = None
        self.set_model("lr")  # LogisticRegression

    @property
    def model_is_set(self) -> bool:
        return self._model is not None

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

        data = data.astype({k: v for k, v in FEAT_DTYPES.items() if k in data.columns and v != "DATETIME"})

        # Getting dummies
        str_cols = {k: v for k, v in FEAT_DTYPES.items() if k in data.columns and v == "str"}
        if str_cols:
            if self.dummies is None:
                assert not self.model_is_fitted
                data = pd.concat(
                    [data.drop(str_cols, axis=1)] + [pd.get_dummies(data[ft].astype(str), prefix=ft) for ft in str_cols],
                    axis=1
                )
                self.dummies = {
                    ft: [c[len(ft)+1:] for c in data.columns if c.startswith(f"{ft}_")]
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
                        "Values used must be contained inside the model's fitted dummy columns"
                    )

                data = pd.concat(
                    [data.drop(str_cols, axis=1)] + [pd.get_dummies(data[ft].astype(str), prefix=ft) for ft in str_cols],
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
                        [data] + [pd.DataFrame(np.zeros((data.shape[0], len(missing_cols)), dtype=int), columns=missing_cols)],
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
        model_type: Literal["xgb", "lr"],
        params: Optional[dict] = None
    ) -> None:
        assert not self.model_is_set
        _params = params if params is not None else {}

        if model_type == "xgb":
            self._model = XGBClassifier(**_params)
        elif model_type == "lr":
            self._model = LogisticRegression(**_params)
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
                "Wrong feature names"  # if the API was private we could have more verbosity here
            )

        cond = all(
            VALIDATORS[feat](fl[feat])
            for feat in self.input_cols
            for fl in flights
        )
        if not cond:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Wrong feature values"  # if the API was private we could have more verbosity here
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
        self._model = None
        print("Model has been unset")

    @staticmethod
    def process_delay(data: pd.DataFrame) -> pd.DataFrame:
        min_diff = data.apply(get_min_diff, axis=1)
        data = data.drop(["Fecha-I", "Fecha-O"], axis=1)
        data["delay"] = (min_diff > DELAY_THRESH_IN_MINS).astype(int)
        return data


# def train_xgboost(
#     x_train: pd.DataFrame,
#     x_test: pd.DataFrame,
#     y_train: np.ndarray,
#     y_test: np.ndarray
# ):
#     xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
#     xgb_model.fit(x_train, y_train)
#     xgboost_y_preds = xgb_model.predict(x_test)
#     xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]
#     confusion_matrix(y_test, xgboost_y_preds)
#     print(classification_report(y_test, xgboost_y_preds))


# def train_logistic_regression(
#     x_train: pd.DataFrame,
#     x_test: pd.DataFrame,
#     y_train: np.ndarray,
#     y_test: np.ndarray
# ):
#     reg_model = LogisticRegression()
#     reg_model.fit(x_train, y_train)
#     reg_y_preds = reg_model.predict(x_test)
#     confusion_matrix(y_test, reg_y_preds)
#     print(classification_report(y_test, reg_y_preds))
