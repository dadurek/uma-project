import numpy as np
import pandas as pd
from enum import Enum

ROUNDING_AMOUNT = 3


class Type(Enum):
    ROOT = 'root'
    LEFT = 'left'
    RIGHT = 'right'


class Node:

    def __init__(
            self,
            X: pd.core.frame.DataFrame,
            Y: list,
            max_depth: int,
            min_elements: int,
            depth=None,
            type=None,
            x_mean=None,
            feature=None
    ):
        self.X = X
        self.Y = Y
        self.max_depth = max_depth
        self.min_elements = min_elements
        self.depth = depth if depth else 0
        self.elements = len(Y)
        self.features = self.X.columns.tolist()
        self.mse = self.get_mse(self.Y)

        self.y_mean = np.mean(self.Y)
        self.x_mean = x_mean if x_mean else 0
        self.feature = feature if feature else ''
        self.type = type if type else Type.ROOT

        self.left = None
        self.right = None

    @staticmethod
    def get_mse(y: list) -> float:
        y_mean = np.mean(y)
        elements = len(y)
        residuals = (y - y_mean) ** 2
        return np.sum(residuals) / elements

    @staticmethod
    def get_mse_for_two_vectors(left_y: list, right_y: list) -> float:
        left_mean = np.mean(left_y)
        right_mean = np.mean(right_y)
        residuals_left = left_y - left_mean
        residuals_right = right_y - right_mean
        residuals = np.concatenate((residuals_left, residuals_right), axis=None)
        elements = len(residuals)
        residuals = residuals ** 2
        return np.sum(residuals) / elements

    @staticmethod
    def roulette(mses: list) -> int:  # return index of best
        fitness = []
        probability = []
        for mse in mses:
            fitness.append(1 / (1 + mse))
        s = np.nansum(fitness)
        for f in fitness:
            probability.append(f / s)
        return np.argmax(probability)

    @staticmethod
    def neighbours_mean(x: list) -> list:  # TODO make it bulletproof
        x.sort()
        x_means = []
        for i in range(0, len(x) - 1):
            x_means.append(np.mean([x[i], x[i + 1]]))
        return x_means

    def best_split(self) -> tuple:
        d = self.X.copy()
        d['Y'] = self.Y
        mses = []

        for feature in self.features:
            x_means = self.neighbours_mean(d[feature].to_list())
            for x in x_means:
                left_y = d[d[feature] < x]['Y'].values
                right_y = d[d[feature] >= x]['Y'].values
                mse = self.get_mse_for_two_vectors(left_y, right_y)
                mses.append(mse)
        best_index = self.roulette(mses)

        # length of x_means, because x_mean is vector means of neighbours in X vector, so it's one element less than X or Y
        size_x_means = len(d['Y'].to_list()) - 1
        feature_index = int(best_index / size_x_means)
        feature = self.features[feature_index]
        x_mean_index = int(best_index % size_x_means)
        x_mean = self.neighbours_mean(d[feature].to_list())[x_mean_index]

        return feature, x_mean

    def grow_tree(self):
        df = self.X.copy()
        df['Y'] = self.Y
        if self.depth < self.max_depth and self.elements >= self.min_elements:
            feature, x_mean = self.best_split()

            left_df = df[df[feature] <= x_mean].copy()
            right_df = df[df[feature] > x_mean].copy()
            left = Node(
                X=left_df[self.features],
                Y=left_df['Y'].values.tolist(),
                max_depth=self.max_depth,
                min_elements=self.min_elements,
                depth=self.depth + 1,
                type=Type.LEFT,
                x_mean=x_mean,
                feature=feature
            )

            self.left = left
            self.left.grow_tree()

            right = Node(
                right_df[self.features],
                right_df['Y'].values.tolist(),
                max_depth=self.max_depth,
                min_elements=self.min_elements,
                depth=self.depth + 1,
                type=Type.RIGHT,
                x_mean=x_mean,
                feature=feature
            )

            self.right = right
            self.right.grow_tree()

    def print_node(self, width=3):
        const = int(self.depth * width ** 2)
        padding = "-" * const

        if self.type is Type.ROOT:
            print(self.type.value)
        else:
            if self.type is Type.LEFT:
                print(f"|{padding} Split rule: {self.feature} <= {round(self.x_mean, ROUNDING_AMOUNT)}")
            else:
                print(f"|{padding} Split rule: {self.feature} > {round(self.x_mean, ROUNDING_AMOUNT)}")

        print(f"{' ' * const}   | Type : {self.type.value}")
        print(f"{' ' * const}   | MSE of the node : {round(self.mse, ROUNDING_AMOUNT)}")
        print(f"{' ' * const}   | Count of observations in node : {self.elements}")
        print(f"{' ' * const}   | Prediction of node : {round(self.y_mean, ROUNDING_AMOUNT)}")
        print(f"{' ' * const}   | Remaining elements : {self.elements}")
        print(f"{' ' * const}   | x_mean value : {self.x_mean}")

    def print_tree(self):
        self.print_node()

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()

    def predict(self, df: pd.core.frame.DataFrame, new_column_name: str) -> pd.core.frame.DataFrame:
        df[new_column_name] = 0
        for index, row in df.iterrows():
            value = self.recursive(row=row)
            df[new_column_name][index] = value
        return df

    def recursive(self, row: pd.core.series.Series):
        while (self.left is not None) and (self.right is not None):
            feature = self.left.feature if self.left.feature else self.right.feature  # pick left feature, if it does not exist pick feature from right
            x_mean = self.left.x_mean if self.left.x_mean else self.right.x_mean  # pick left x_mean, if it does not exist pick x_mean from right
            if row[feature] <= x_mean:  # left
                return self.left.recursive(row=row)
            elif row[feature] > x_mean:  # right
                return self.right.recursive(row=row)
        return self.y_mean
