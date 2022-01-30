from helpers import *

if __name__ == '__main__':
    quantity_from_csv = 8124
    max_depth = 10
    min_elements = 5
    values = [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]

    file_path = "datasets/agaricus-lepiota.data"
    columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
               "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
               "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
               "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    to_estimate_column_name = "class"
    features_columns_name = columns[1:]
    predicted_values_column_name = 'prediction'
    approx_value_column_name = 'approx_value'

    dataFrame = prepare_data_frame_mushrooms(file_path=file_path, columns_name=columns, size=quantity_from_csv)

    dataFrame.sample(frac=1)
    dataFrame.sample(frac=1)

    roulette = test_errors_mushrooms(
        values,
        True,
        dataFrame,
        max_depth,
        min_elements,
        to_estimate_column_name,
        features_columns_name,
        predicted_values_column_name,
        approx_value_column_name
    )

    normal = test_errors_mushrooms(
        values,
        False,
        dataFrame,
        max_depth,
        min_elements,
        to_estimate_column_name,
        features_columns_name,
        predicted_values_column_name,
        approx_value_column_name
    )

    print_pretty(values, roulette, normal)


