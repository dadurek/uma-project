from helpers import *

if __name__ == '__main__':
    quantity_from_csv = 1000
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

    ###############################
    # df = df.head(quantity_from_csv)
    ###############################

    for index, row in df.iterrows():
        for column, value in row.items():
            df[column][index] = ord(value)  # change char to value in utf-8

    data_frame = df.head(800)

    X = data_frame[features_columns_name]
    Y = data_frame[to_estimate_column_name].values.tolist()

    tree = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)
    tree.configure(roulette_option=False, rounding_amount=3, width_print=9)
    tree.grow()
    tree.print_tree()

    dataFrame = df.tail(200)

    dataFrame = tree.predict(df=dataFrame, new_column_name=predicted_values_column_name)

    error = compute_error_mushroom(df=dataFrame, true_value=to_estimate_column_name, predicted=predicted_values_column_name, approx_value=approx_value_column_name)

    for index, row in dataFrame.iterrows():
        print("real value: " + str(row[to_estimate_column_name]))
        print("estimated value: " + str(row[predicted_values_column_name]))
        print("approxed value: " + str(row[approx_value_column_name]))
        print("-------------------")

    print(f"ERROR: {error * 100}%")

