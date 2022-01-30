from helpers import *

if __name__ == '__main__':
    max_depth = 3
    min_elements = 3

    file_path = "datasets/agaricus-lepiota.data"
    columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
               "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
               "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
               "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    to_estimate_column_name = "class"
    features_columns_name = columns[1:]
    predicted_values_column_name = 'prediction'
    approx_value_column_name = 'approx_value'

    df = pd.read_csv(file_path, header=None)
    df.columns = columns

    col_map = {}

    for column_name in columns:
        unique_values = list(set(df[column_name]))
        unique_values.sort()
        col_map[column_name] = unique_values
        dictionary = {unique_values[i]: i for i in range(0, len(unique_values))}
        for index, value in df[column_name].items():
            df[column_name][index] = dictionary[value]

    data_frame = df.head(1000)

    X = data_frame[features_columns_name]
    Y = data_frame[to_estimate_column_name].values.tolist()

    tree = Node(X=X,
                Y=Y,
                max_depth=max_depth,
                min_elements=min_elements)
    tree.configure(roulette_option=False,
                   rounding_amount=3,
                   width_print=9)
    tree.grow()
    tree.print_tree()

    dataFrame = df.tail(500)

    dataFrame = tree.predict(df=dataFrame,
                             new_column_name=predicted_values_column_name)

    confusion_matrix = compute_confusion_matrix(df=dataFrame,
                                                true_value=to_estimate_column_name,
                                                predicted=predicted_values_column_name,
                                                approx_value=approx_value_column_name)
    print(confusion_matrix)

    error = compute_error_using_confusion_matrix(confusion_matrix)

    print(f"ERROR: {error * 100}%")
