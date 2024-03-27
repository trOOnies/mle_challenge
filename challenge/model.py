import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Optional, Tuple, Union, List, Literal
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier



class DelayModel:
    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

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
        return

    def set_model(
        self,
        model_type: Literal["xgb", "lr"],
        params: Optional[dict] = None
    ) -> None:
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
        self._model.fit(features, target)

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
        return self._model.predict(features).tolist()


xgb_clf = DelayModel()
xgb_clf.set_model("xgb", params={"random_state": 1, "learning_rate": 0.01})

lr_clf = DelayModel()
lr_clf.set_model("lr")


# def train_xgboost(
#     x_train: pd.DataFrame,
#     x_test: pd.DataFrame,
#     y_train: "ndarray",
#     y_test: "ndarray"
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
#     y_train: "ndarray",
#     y_test: "ndarray"
# ):
#     reg_model = LogisticRegression()
#     reg_model.fit(x_train, y_train)
#     reg_y_preds = reg_model.predict(x_test)
#     confusion_matrix(y_test, reg_y_preds)
#     print(classification_report(y_test, reg_y_preds))
