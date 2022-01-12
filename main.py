from regressionTree import *
from helpers import *


if __name__ == '__main__':
    quantity_from_csv = 1000
    max_depth = 3
    min_elements = 3

    file_path = "housing.csv"
    to_estimate_column_name = "median_house_value"
    features_columns_name = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                'households', 'median_income', 'ocean_proximity']
    predicted_values_column_name = 'prediction'

    dataFrame = prepare_data_frame(file_path=file_path, columns_name=np.concatenate((to_estimate_column_name, features_columns_name), axis=None), size=quantity_from_csv)

    X, Y = prepare_data(df=dataFrame, to_estimate=to_estimate_column_name, features=features_columns_name)

    tree = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)
    tree.configure(roulette_option=True)
    tree.grow()
    tree.print_tree()

    dataFrame = dataFrame.head(50)

    tree.predict(df=dataFrame, new_column_name=predicted_values_column_name)

    for index, row in dataFrame.iterrows():
        print("real value: " + str(row[to_estimate_column_name]))
        print("estimated value: " + str(row[predicted_values_column_name]))
        print("-------------------")

    error = compute_error(df=dataFrame, true_value=to_estimate_column_name, predicted=predicted_values_column_name)

    print(f"ERROR: {error * 100}%")
