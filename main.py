from regressionTree import *
from prepareData import *


if __name__ == '__main__':
    quantity_from_csv = 100
    max_depth = 3
    min_elements = 3

    file_path = "housing.csv"
    to_estimate = "median_house_value"
    features = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                'households', 'median_income', 'ocean_proximity']

    dataFrame = prepare_data_frame(file_path=file_path, columns_name=np.concatenate((to_estimate, features), axis=None), size=quantity_from_csv)

    X, Y = prepare_data(df=dataFrame, to_estimate=to_estimate, features=features)

    tree = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)

    tree.configure(roulette_option=True)  # enable roulette, if false in function best_split() picked will be value with highest probability
    tree.configure(rounding_amount=4) # amount of rounding of values when printing tree
    tree.configure(width_print=9) # spaces padding when printing tree
    tree.configure(roulette_option=True, rounding_amount=4, width_print=9)

    tree.grow()
    tree.print_tree()

    dataFrame = dataFrame.head(50)

    new_column_name = 'estimated'

    tree.predict(df=dataFrame, new_column_name=new_column_name)

    for index, row in dataFrame.iterrows():
        print("real value: " + str(row[to_estimate]))
        print("estimated value: " + str(row[new_column_name]))
        print("-------------------")
