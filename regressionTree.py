import numpy as np
import pandas as pd
import random as rand
from enum import Enum


class Type(Enum):
    ROOT = 'root'
    LEFT = 'left'
    RIGHT = 'right'


class Node:
    ROUNDING_AMOUNT = 3
    WIDTH_PRINT = 9
    ROULETTE_ENABLED = True

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
        self.length_elements = len(Y)
        self.features = self.X.columns.tolist()
        self.mse = self.__get_mse(self.Y)
        self.y_mean = np.mean(self.Y)
        self.x_mean = x_mean if x_mean else 0
        self.feature = feature if feature else ''
        self.type = type if type else Type.ROOT

        self.left = None
        self.right = None

    @staticmethod
    def configure(roulette_option=ROULETTE_ENABLED, rounding_amount=ROUNDING_AMOUNT, width_print=WIDTH_PRINT):
        """configuration function, provide option to change settings used in this module"""
        Node.ROULETTE_ENABLED = roulette_option
        Node.ROUNDING_AMOUNT = rounding_amount
        Node.WIDTH_PRINT = width_print

    @staticmethod
    def __get_mse(y: list) -> float:
        """calculate mse of list"""
        y_mean = np.nanmean(y)
        elements = len(y)
        residuals = (y - y_mean) ** 2
        return np.sum(residuals) / elements

    @staticmethod
    def __get_mse_for_two_vectors(left_y: list, right_y: list) -> float:
        """calculate mse of two lists"""
        left_mean = 0
        right_mean = 0
        if len(left_y):
            left_mean = np.nanmean(left_y)
        if len(right_y):
            right_mean = np.nanmean(right_y)
        residuals_left = left_y - left_mean
        residuals_right = right_y - right_mean
        residuals = np.concatenate((residuals_left, residuals_right), axis=None)
        elements = len(residuals)
        residuals = residuals ** 2
        return np.sum(residuals) / elements

    def __pick_mse(self, mses: list) -> int:  # return index of picked mse
        """return index of picked mse, based on value ROULETTE_ENABLED return by roulette or highest probability"""
        fitness = []
        probability = []
        for mse in mses:
            fitness.append(1 / (1 + mse))
        s = np.nansum(fitness)
        for f in fitness:
            probability.append(f / s)

        if Node.ROULETTE_ENABLED:
            picked_value = rand.choices(population=mses, weights=probability)
            return mses.index(picked_value)
        else:
            return np.argmax(probability)

    @staticmethod
    def __neighbours_mean(x: list) -> list:
        """return list of neighbours from provided list"""
        x.sort()
        x = list(set(x)) # only unique values
        x_means = []
        for i in range(0, len(x) - 1):
            x_means.append(np.mean([x[i], x[i + 1]]))
        return x_means

    def __split(self) -> tuple:
        """decide which feature use to split X vector and provide value to split"""
        d = self.X.copy()
        d['Y'] = self.Y
        mses = []

        for feature in self.features:
            x_means = self.__neighbours_mean(d[feature].to_list())
            for x in x_means:
                left_y = d[d[feature] < x]['Y'].values
                right_y = d[d[feature] >= x]['Y'].values
                mse = self.__get_mse_for_two_vectors(left_y, right_y)
                mses.append(mse)

        best_index = self.__pick_mse(mses)
        for feature in self.features:
            x_means = self.__neighbours_mean(d[feature].to_list())
            size = len(x_means)
            if best_index - size < 0:
                return feature, x_means[best_index]
            else:
                best_index = best_index - size

    def grow(self):
        """grow regression tree"""
        df = self.X.copy()
        df['Y'] = self.Y
        if self.depth < self.max_depth and self.length_elements >= self.min_elements:
            feature, x_mean = self.__split()

            left_df = df[df[feature] <= x_mean].copy()
            right_df = df[df[feature] > x_mean].copy()

            if len(left_df['Y']) >= self.min_elements:
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
                self.left.grow()

            if len(right_df['Y']) >= self.min_elements:
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
                self.right.grow()

    def predict(self, df: pd.core.frame.DataFrame, new_column_name: str) -> pd.core.frame.DataFrame:
        """predict value based on generated tree"""
        df[new_column_name] = 0
        for index, row in df.iterrows():
            value = self.__recursive_y_mean_search(row=row)
            df[new_column_name][index] = value
        return df

    def __recursive_y_mean_search(self, row: pd.core.series.Series):
        """recursive search of y_mean"""
        while self.left is not None and self.right is not None:
            feature = self.left.feature if self.left.feature else self.right.feature  # pick left feature, if it does not exist pick feature from right
            x_mean = self.left.x_mean if self.left.x_mean else self.right.x_mean  # pick left x_mean, if it does not exist pick x_mean from right
            if row[feature] <= x_mean:  # left
                return self.left.__recursive_y_mean_search(row=row)
            elif row[feature] > x_mean:  # right
                return self.right.__recursive_y_mean_search(row=row)
        return self.y_mean

    def __print_node(self):
        """print node based on type od node"""
        const = int(self.depth * Node.WIDTH_PRINT)
        padding = "-" * const

        if self.type is Type.ROOT:
            print(f"Start : {self.type.value}")
        else:
            if self.type is Type.LEFT:
                print(f"|{padding} Split rule: {self.feature} <= {round(self.x_mean, self.ROUNDING_AMOUNT)}")
            else:
                print(f"|{padding} Split rule: {self.feature} > {round(self.x_mean, self.ROUNDING_AMOUNT)}")

        print(f"{' ' * const}   | Type : {self.type.value}")
        print(f"{' ' * const}   | MSE of the node : {round(self.mse, self.ROUNDING_AMOUNT)}")
        print(f"{' ' * const}   | Count of observations in node : {self.length_elements}")
        print(f"{' ' * const}   | Prediction of node : {round(self.y_mean, self.ROUNDING_AMOUNT)}")
        print(f"{' ' * const}   | Remaining elements : {self.length_elements}")
        print(f"{' ' * const}   | x_mean value : {self.x_mean}")

    def print_tree(self):
        """recursive function of printing tree"""
        self.__print_node()

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()
