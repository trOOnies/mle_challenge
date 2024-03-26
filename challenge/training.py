import pandas as pd
from typing import TYPE_CHECKING, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
if TYPE_CHECKING:
    from numpy import ndarray


def get_splits(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, "ndarray", "ndarray"]:
    training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)
    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
    )
    target = data['delay']
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)
    # print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
    # y_train.value_counts('%')*100
    # y_test.value_counts('%')*100
    return x_train, x_test, y_train, y_test


def train_xgboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: "ndarray",
    y_test: "ndarray"
):
    xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    xgb_model.fit(x_train, y_train)
    xgboost_y_preds = xgb_model.predict(x_test)
    xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]
    confusion_matrix(y_test, xgboost_y_preds)
    print(classification_report(y_test, xgboost_y_preds))


def train_logistic_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: "ndarray",
    y_test: "ndarray"
):
    reg_model = LogisticRegression()
    reg_model.fit(x_train, y_train)
    reg_y_preds = reg_model.predict(x_test)
    confusion_matrix(y_test, reg_y_preds)
    print(classification_report(y_test, reg_y_preds))
