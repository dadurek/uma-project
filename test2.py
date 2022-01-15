from regressionTree import *
from helpers import *
from regressionTree import *
from helpers import *
import time
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
    min_elements = 3
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
    result_depth_time = []

    for max_depth in range(1,21,1):
        start = time.perf_counter()
        print("Computing for tree with max_depth = "+str(max_depth))
        result_depth.append(get_error(dataFrame.head(10000),dataFrame.tail(2000),True,max_depth,min_elements))
        print("Got error = "+str(result_depth[-1]))
        end = time.perf_counter()
        result_depth_time.append(end-start)
        print("Got time = " + str(result_depth_time[-1]))


    max_depth = 30
    print(result_depth)
    print(result_depth_time)
    df = pd.DataFrame(result_depth)
    df.to_csv('result_depth.csv')
    df = pd.DataFrame(result_depth_time)
    df.to_csv('result_depth_time.csv')

    result_elements = []
    result_elements_time = []
    for max_elements in range(2,103,10):
        start = time.perf_counter()
        print("Computing for tree with min_elements = "+str(max_elements))
        result_elements.append(get_error(dataFrame.head(10000),dataFrame.tail(2000),True,max_depth,max_elements))
        print("Got error = "+str(result_elements[-1]))
        end = time.perf_counter()
        result_elements_time.append(end-start)

    print(result_elements)
    print(result_elements_time)
    df = pd.DataFrame(result_elements)
    df.to_csv('result_elements.csv')
    df = pd.DataFrame(result_elements_time)
    df.to_csv('result_elements_time.csv')



