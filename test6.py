from helpers import *

if __name__ == '__main__':
    quantity_from_csv = 8124
    max_depth = 30

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

    result_elements = []
    result_elements_time = []
    for min_elements in range(2, 103, 10):
        start = time.perf_counter()
        print("Computing for tree with min_elements = " + str(min_elements))
        result_elements.append(
            get_error_mushrooms(
                dataFrame.head(7000),
                dataFrame.tail(1000),
                True,
                max_depth,
                min_elements,
                to_estimate_column_name,
                features_columns_name,
                predicted_values_column_name,
                approx_value_column_name
            )
        )
        print("Got error = " + str(result_elements[-1]))
        end = time.perf_counter()
        result_elements_time.append(end - start)

    print(result_elements)
    print(result_elements_time)
    df = pd.DataFrame(result_elements)
    df.to_csv('mushrooms_result_elements.csv')
    df = pd.DataFrame(result_elements_time)
    df.to_csv('mushrooms_result_elements_time.csv')


