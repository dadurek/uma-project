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


def test_errors(values,roulette, dataFrame,max_depth,min_elements):
    list = []
    for value in values:
        print("Computing for tree with roulette = "+str(roulette)+" and number of elements = "+str(value))
        list.append(get_error(dataFrame.head(value), dataFrame.tail(2000), roulette, max_depth, min_elements))
    return list

def print_pretty(values,roulette,normal):
    for value_r,value_n,value in zip(roulette, normal,values):
        lane_1 = "Teaching with number of elements = "+str(value)+" "
        print(lane_1)
        print(len(lane_1)*"-")
        print("Error for roulette = "+str(value_r))
        print("Error for normal = " + str(value_n))
        print(len(lane_1) * "-")
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    quantity_from_csv = 20000
    max_depth = 10
    min_elements = 5
    values = [10,50,100,200,500,1000,2000,3000,4000]

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

    roulette = test_errors(values,True, dataFrame,max_depth,min_elements)
    normal = test_errors(values,False, dataFrame,max_depth,min_elements)

    print_pretty(values,roulette,normal)
