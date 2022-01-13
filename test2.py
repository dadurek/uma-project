from regressionTree import *
from helpers import *
from regressionTree import *
from helpers import *
import warnings

def get_error(train_df, validate_df, roulette,max_depth,min_elements):
    X, Y = prepare_data(df=train_df, to_estimate=to_estimate_column_name, features=features_columns_name)

    tree = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)
    tree.configure(roulette_option=roulette)
    tree.grow()

    tree.predict(df=validate_df, new_column_name=predicted_values_column_name)

    error = compute_error(df=validate_df, true_value=to_estimate_column_name, predicted=predicted_values_column_name)
    return error

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    quantity_from_csv = 20000
    min_elements = 5
    max_depth = 10

    file_path = "housing.csv"
    to_estimate_column_name = "median_house_value"
    features_columns_name = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                             'households', 'median_income', 'ocean_proximity']
    predicted_values_column_name = 'prediction'

    dataFrame = prepare_data_frame(file_path=file_path,
                                   columns_name=np.concatenate((to_estimate_column_name, features_columns_name),
                                                               axis=None), size=quantity_from_csv)

    dataFrame.sample(frac=1)
    dataFrame.sample(frac=1)

    result_depth = []

    for max_depth in range(1,32,3):
        print("Computing for tree with max_depth = "+str(max_depth))
        result_depth.append(get_error(dataFrame.head(500),dataFrame.tail(2000),True,max_depth,min_elements))
        print("Got error = "+str(result_depth[-1]))

    print(result_depth)

    result_elements = []
    for max_elements in range(1,102,10):
        print("Computing for tree with max_depth = "+str(max_depth))
        result_depth.append(get_error(dataFrame.head(500),dataFrame.tail(2000),True,max_depth,min_elements))
        print("Got error = "+str(result_depth[-1]))

    print(result_elements)



