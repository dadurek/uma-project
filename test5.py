from helpers import *

if __name__ == '__main__':
    quantity_from_csv = 10000
    min_elements = 5

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

    result_depth = []
    result_depth_time = []

    for max_depth in range(1, 21, 1):
        start = time.perf_counter()
        print("Computing for tree with max_depth = " + str(max_depth))
        result_depth.append(
            get_error_mushrooms(
                dataFrame.head(10000),
                dataFrame.tail(2000),
                True,
                max_depth,
                min_elements,
                to_estimate_column_name,
                features_columns_name,
                predicted_values_column_name,
                approx_value_column_name
            )
        )
        print("Got error = " + str(result_depth[-1]))
        end = time.perf_counter()
        result_depth_time.append(end - start)
        print("Got time = " + str(result_depth_time[-1]))

    max_depth = 30
    print(result_depth)
    print(result_depth_time)
    df = pd.DataFrame(result_depth)
    df.to_csv('mushrooms_result_depth.csv')
    df = pd.DataFrame(result_depth_time)
    df.to_csv('mushrooms_result_depth_time.csv')



