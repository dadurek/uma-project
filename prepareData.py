import pandas as pd


def prepare_data_frame(file_path: str, columns_name: list, size: int) -> pd.core.frame.DataFrame:
    df = pd.read_csv(file_path)
    ocean_proximity_dict = {'NEAR BAY': 1, '<1H OCEAN': 2, 'INLAND': 3, 'NEAR OCEAN': 4, 'ISLAND': 5}

    # cast ocean_proximity to numeric number
    if 'ocean_proximity' in columns_name:
        df['ocean_proximity'] = pd.Series(ocean_proximity_dict[i] for i in df['ocean_proximity'])

    # drop na values for both estimating value and features
    df.dropna(subset=columns_name, inplace=True)

    # cast to numeric
    for cn in columns_name:
        df[cn] = pd.to_numeric(df[cn])

    # always pick random sample with
    df = df.head(size)  # TODO decide if we use head() or sample()

    return df


def prepare_data(df: pd.core.frame.DataFrame, to_estimate: str, features: list, ) -> tuple:
    # set of features
    X = df[features]

    # continuous variable
    Y = df[to_estimate].values.tolist()

    return X, Y
