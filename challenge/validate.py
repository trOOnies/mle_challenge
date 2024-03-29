import yaml
from pandas import to_datetime
from typing import Any, Tuple, Dict
from challenge.classes import Validator, ValidatorDict, FeatureName


def get_dtype_checker(dtype) -> Validator:
    def clb(v: Any) -> bool:
        return isinstance(dtype(v), dtype)
    return clb


def get_validator(feat: str, vals) -> Tuple[Validator, str]:
    if vals == "str":
        return get_dtype_checker(str), "str"
    elif vals == "int":
        return get_dtype_checker(int), "int"
    elif vals == "uint":
        def clb(v: Any) -> bool:
            return isinstance(v, int) and v >= 0
        return clb, "int"
    elif vals == "datetime":
        def clb(v: Any) -> bool:
            try:
                to_datetime(v)
            except:
                return False
            else:
                return True
        return clb, "DATETIME"
    elif isinstance(vals, list):
        assert vals
        dtype = type(vals[0])
        assert all(type(val) == dtype for val in vals)
        if dtype == str:
            str_dtype = "str"
        elif dtype == int:
            str_dtype = "int"
        else:
            raise NotImplementedError(f"dtype '{str_dtype}' not implemented.")

        def clb(v: Any) -> bool:
            return isinstance(v, dtype) and v in vals
        return clb, str_dtype
    elif isinstance(vals, dict):
        assert isinstance(vals["dtype"], str) and vals["dtype"] == "int"
        assert isinstance(vals["from"], int) and isinstance(vals["to"], int)
        assert vals["from"] <= vals["to"]
        if vals["dtype"] == "str":
            str_dtype = "str"
        elif vals["dtype"] == "int":
            str_dtype = "int"
        else:
            raise NotImplementedError(f"dtype '{str_dtype}' not implemented.")

        def clb(v: Any) -> bool:
            return isinstance(v, int) and (vals["from"] <= v) and (v <= vals["to"])
        return clb, str_dtype
    else:
        raise ValueError(f"Feature {feat} validators couldn't be understood")


def get_validators() -> Tuple[ValidatorDict, Dict[FeatureName, str]]:
    print("Getting validators...")
    with open("challenge/features.yaml", "r") as f:
        validators = yaml.safe_load(f)

    validators = {
        feat: get_validator(feat, vals)
        for feat, vals in validators.items()
    }
    dtypes = {k: vs[1] for k, vs in validators.items()}
    validators = {k: vs[0] for k, vs in validators.items()}
    print("Validators set.")

    return validators, dtypes
