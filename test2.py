from helpers import *
from regressionTree import *

if __name__ == '__main__':
    quantity_from_csv = 10000
    min_elements = 5
    file_path = "datasets/housing.csv"
    to_estimate_column_name = "median_house_value"
    features_columns_name = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                             'households', 'median_income', 'ocean_proximity']
    predicted_values_column_name = 'prediction'

    dataFrame = prepare_data_frame_housing(
        file_path=file_path,
        columns_name=np.concatenate((to_estimate_column_name, features_columns_name), axis=None),
        size=quantity_from_csv
    )

    dataFrame.sample(frac=1)
    dataFrame.sample(frac=1)

    result_depth = []
    result_depth_time = []

    for max_depth in range(1, 21, 1):
        start = time.perf_counter()
        print("Computing for tree with max_depth = " + str(max_depth))
        result_depth.append(
            get_error_housing(
                dataFrame.head(10000),
                dataFrame.tail(2000),
                True,
                max_depth,
                min_elements,
                to_estimate_column_name,
                features_columns_name,
                predicted_values_column_name
            )
        )
        print("Got error = " + str(result_depth[-1]))
        end = time.perf_counter()
        result_depth_time.append(end - start)
        print("Got time = " + str(result_depth_time[-1]))

    df = pd.DataFrame(result_depth)
    df.to_csv('housing_result_depth.csv')
    df = pd.DataFrame(result_depth_time)
    df.to_csv('housing_result_depth_time.csv')
