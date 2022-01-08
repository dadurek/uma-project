import numpy as np
import pandas as pd


class Node():

    def __init__(
            self,
            X: pd.core.frame.DataFrame,
            Y: list,
            max_depth: int,
            min_elements: int,
            depth=None
    ):
        self.X = X
        self.Y = Y
        self.max_depth = max_depth
        self.min_elements = min_elements
        self.depth = depth if depth else 0
        self.elements = len(Y)
        self.features = self.X.columns.tolist()
        self.mse = self.get_mse(self.Y)
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
        s = sum(fitness)
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

    def grow_tree(self):
        df = self.X.copy()
        df['Y'] = self.Y
        if self.depth < self.max_depth and self.elements >= self.min_elements:
            best_feature, best_value = self.best_split()

            left_df = df[df[best_feature] <= best_value].copy()
            right_df = df[df[best_feature] > best_value].copy()
            left = Node(
                X=left_df[self.features], # in next step of tree we should remove current best_feature??????
                Y=left_df['Y'].values.tolist(),
                max_depth=self.max_depth,
                min_elements=self.min_elements,
                depth=self.depth + 1
            )

            self.left = left
            self.left.grow_tree()

            right = Node(
                right_df[self.features], # in next step of tree we should remove current best_feature??????
                right_df['Y'].values.tolist(),
                max_depth=self.max_depth,
                min_elements=self.min_elements,
                depth=self.depth + 1,
            )

            self.right = right
            self.right.grow_tree()

    def best_split(self) -> tuple:
        df = self.X.copy()
        df['Y'] = self.Y
        mses = []

        for feature in self.features:
            x_means = self.neighbours_mean(df[feature].to_list())
            for x in x_means:
                left_y = df[df[feature] < x]['Y'].values
                right_y = df[df[feature] >= x]['Y'].values
                mse = self.get_mse_for_two_vectors(left_y, right_y)
                mses.append(mse)

        best_index = self.roulette(mses)
        for feature in self.features:
            x_mean = self.neighbours_mean(df[feature].to_list())[best_index]
            left_y = df[df[feature] < x_mean]['Y'].values
            right_y = df[df[feature] >= x_mean]['Y'].values
            mse = self.get_mse_for_two_vectors(left_y, right_y)
            if mse == mses[best_index]:
                return feature, mses[best_index]

        return None, None

if __name__ == '__main__':
    quantity_from_csv = 1000
    max_depth = 3
    min_elements = 3
    file_path = "housing.csv"
    to_estimate = "median_house_value"
    # features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
    #             'households', 'median_income', 'ocean_proximity']
    features = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                'households', 'median_income', 'ocean_proximity']
    ocean_proximity_dict = {'NEAR BAY': 1, '<1H OCEAN': 2, 'INLAND': 3, 'NEAR OCEAN': 4, 'ISLAND': 5}

    df = pd.read_csv(file_path)

    # cast ocean_proximity to numeric number
    df[features[-1]] = pd.Series(ocean_proximity_dict[i] for i in df[features[-1]])

    # cast to numeric
    for ft in features:
        df[ft] = pd.to_numeric(df[ft])

    # pick defined quantity TODO maybe mix elements here as they are in order of  longitude
    df = df.head(quantity_from_csv)
    X = df[features]  # set of features
    Y = df[to_estimate].values.tolist()  # continous variable

    root = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)

    root.grow_tree()
