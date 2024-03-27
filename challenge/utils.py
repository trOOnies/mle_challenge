import pandas as pd


def get_rate_from_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    delays = {k: 0 for k in data[column].unique()}
    delayed_rows = data[column][data.delay == 1]
    delays.update(delayed_rows.value_counts().to_dict())
    del delayed_rows

    total = data[column].value_counts().to_dict()

    rates = {
        name: (
            round(total / delays[name], 2)
            if delays[name] > 0
            else 0
        )
        for name, total in total.items()
    }

    return pd.DataFrame.from_dict(
        rates,
        orient='index',
        columns=['Tasa (%)']
    )
