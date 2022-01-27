import time

import numpy as np
import pandas as pd

from regressionTree import Node


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
    df = df.head(size)

    return df


def prepare_data(df: pd.core.frame.DataFrame, to_estimate: str, features: list, ) -> tuple:
    # set of features
    X = df[features]

    # continuous variable
    Y = df[to_estimate].values.tolist()

    return X, Y


def compute_error(df: pd.core.frame.DataFrame, true_value: str, predicted: str) -> float:
    vector = (abs((df[true_value] - df[predicted]) / df[true_value])).to_list()
    return np.mean(vector)


def compute_error_mushroom(df: pd.core.frame.DataFrame, true_value: str, predicted: str, approx_value: str) -> float:
    df[approx_value] = 0
    for index, row in df.iterrows():
        predicted_value = row[predicted]
        if abs(predicted_value - ord('e')) < abs(predicted_value - ord('p')):
            df[approx_value][index] = ord('e')
        else:
            df[approx_value][index] = ord('p')
    errors_count = 0
    for index, row in df.iterrows():
        predicted_value = row[approx_value]
        real_value = row[true_value]
        if predicted_value is not real_value:
            errors_count += 1
    return errors_count / len(df)


def get_error(train_df: pd.core.frame.DataFrame, validate_df: pd.core.frame.DataFrame, roulette: bool, max_depth: int,
              min_elements: int, to_estimate_column_name: str, features_columns_name: list,
              predicted_values_column_name: str):
    X, Y = prepare_data(df=train_df, to_estimate=to_estimate_column_name, features=features_columns_name)

    tree = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)
    tree.configure(roulette_option=roulette)
    tree.grow()

    tree.predict(df=validate_df, new_column_name=predicted_values_column_name)

    error = compute_error(df=validate_df, true_value=to_estimate_column_name, predicted=predicted_values_column_name)
    return error


def test_errors(values: list, roulette: bool, dataFrame: pd.core.frame.DataFrame, max_depth: int, min_elements: int,
                to_estimate_column_name: str, features_columns_name: list, predicted_values_column_name: str):
    list_to_return = []
    list_of_time = []
    for value in values:
        print("Computing for tree with roulette = " + str(roulette) + " and number of elements = " + str(value))
        start = time.perf_counter()
        list_to_return.append(
            get_error(
                dataFrame.head(value),
                dataFrame.tail(2000),
                roulette, max_depth,
                min_elements,
                to_estimate_column_name,
                features_columns_name,
                predicted_values_column_name
            )
        )
        end = time.perf_counter()
        print("Got time = " + str(end-start))
        list_of_time.append(end-start)
    list_to_return.append(list_of_time)
    return list_to_return


def print_pretty(values, roulette, normal):
    for value_r, value_n, value in zip(roulette, normal, values):
        lane_1 = "Teaching with number of elements = " + str(value) + " "
        print(lane_1)
        print(len(lane_1) * "-")
        print("Error for roulette = " + str(value_r))
        print("Error for normal = " + str(value_n))
        print(len(lane_1) * "-")
