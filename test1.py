from helpers import *

if __name__ == '__main__':
    quantity_from_csv = 20000
    max_depth = 10
    min_elements = 5
    values = [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000]

    file_path = "datasets/housing.csv"
    to_estimate_column_name = "median_house_value"
    features_columns_name = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                             'households', 'median_income', 'ocean_proximity']
    predicted_values_column_name = 'prediction'

    dataFrame = prepare_data_frame(
        file_path=file_path,
        columns_name=np.concatenate((to_estimate_column_name, features_columns_name), axis=None),
        size=quantity_from_csv
    )

    dataFrame.sample(frac=1)
    dataFrame.sample(frac=1)

    roulette = test_errors(
        values,
        True,
        dataFrame,
        max_depth,
        min_elements,
        to_estimate_column_name,
        features_columns_name,
        predicted_values_column_name
    )

    normal = test_errors(
        values,
        False,
        dataFrame,
        max_depth,
        min_elements,
        to_estimate_column_name,
        features_columns_name,
        predicted_values_column_name
    )

    print_pretty(values, roulette, normal)
