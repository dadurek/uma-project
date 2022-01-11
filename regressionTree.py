import numpy as np
import pandas as pd

ROUNDING_AMOUNT = 3

class Node():

    def __init__(
            self,
            X: pd.core.frame.DataFrame,
            Y: list,
            max_depth: int,
            min_elements: int,
            depth=None,
            node_type=None,
            rule=None
    ):
        self.X = X
        self.Y = Y
        self.max_depth = max_depth
        self.min_elements = min_elements
        self.depth = depth if depth else 0
        self.elements = len(Y)
        self.features = self.X.columns.tolist()
        self.mse = self.get_mse(self.Y)

        self.ymean = np.mean(self.Y)
        self.node_type = node_type if node_type else 'root'
        self.rule = rule if rule else ""

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

    def grow_regression_tree(self):
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
                node_type='left_node',
                rule=f"{feature} <= {round(x_mean, ROUNDING_AMOUNT)}"
            )

            self.left = left
            self.left.grow_regression_tree()

            right = Node(
                right_df[self.features],
                right_df['Y'].values.tolist(),
                max_depth=self.max_depth,
                min_elements=self.min_elements,
                depth=self.depth + 1,
                node_type='right_node',
                rule=f"{feature} > {round(x_mean, ROUNDING_AMOUNT)}"
            )

            self.right = right
            self.right.grow_regression_tree()

    def print_info(self, width=3):
        const = int(self.depth * width ** 2)
        padding = "-" * const

        if self.node_type == 'root':
            print(self.node_type)
        else:
            print(f"|{padding} Split rule: {self.rule}")

        print(f"{' ' * const}   | Type : {self.node_type}")
        print(f"{' ' * const}   | MSE of the node : {round(self.mse, ROUNDING_AMOUNT)}")
        print(f"{' ' * const}   | Count of observations in node : {self.elements}")
        print(f"{' ' * const}   | Prediction of node : {round(self.ymean, ROUNDING_AMOUNT)}")
        print(f"{' ' * const}   | Remaining elements : {self.elements}")

    def print_regression_tree(self):
        self.print_info()

        if self.left is not None:
            self.left.print_regression_tree()

        if self.right is not None:
            self.right.print_regression_tree()


def initialize_tree(X: pd.core.frame.DataFrame,
                    Y: list,
                    max_depth: int,
                    min_elements: int):
    root = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)
    return root


def grow_tree(root: Node):
    root.grow_regression_tree()
    return root


def print_tree(root: Node):
    root.print_regression_tree()


def predict(df: pd.core.frame.DataFrame, root: Node):
    pass
