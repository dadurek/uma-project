from helpers import *
from regressionTree import *

if __name__ == '__main__':
    quantity_from_csv = 10000
    max_depth = 30

    file_path = "datasets/housing.csv"
    to_estimate_column_name = "median_house_value"
    features_columns_name = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                             'households', 'median_income', 'ocean_proximity']
    predicted_values_column_name = 'prediction'

    dataFrame = prepare_data_frame(
        file_path=file_path,
        columns_name=np.concatenate((to_estimate_column_name, features_columns_name), axis=None),
        size=quantity_from_csv
    )

    dataFrame.sample(frac=1)
    dataFrame.sample(frac=1)

    result_elements = []
    result_elements_time = []
    for min_elements in range(2, 103, 10):
        start = time.perf_counter()
        print("Computing for tree with min_elements = " + str(min_elements))
        result_elements.append(
            get_error(
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
        print("Got error = " + str(result_elements[-1]))
        end = time.perf_counter()
        result_elements_time.append(end - start)

    print(result_elements)
    print(result_elements_time)
    df = pd.DataFrame(result_elements)
    df.to_csv('result_elements.csv')
    df = pd.DataFrame(result_elements_time)
    df.to_csv('result_elements_time.csv')
