from regressionTree import *



if __name__ == '__main__':
    # remember about `ocean_proximity` where we need to change string values to numeric

    quantity_from_csv = 1000
    max_depth = 3
    min_elements = 3

    file_path = "housing.csv"
    to_estimate = "median_house_value"
    # features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
    #             'households', 'median_income', 'ocean_proximity']
    features = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                'households', 'median_income', 'ocean_proximity']
    ocean_proximity_dict = {'NEAR BAY': 1, '<1H OCEAN': 2, 'INLAND': 3, 'NEAR OCEAN': 4, 'ISLAND': 5}

    df = pd.read_csv(file_path)

    # cast ocean_proximity to numeric number
    df[features[-1]] = pd.Series(ocean_proximity_dict[i] for i in df[features[-1]])

    df.dropna(subset=features, inplace=True)

    # cast to numeric
    for ft in features:
        df[ft] = pd.to_numeric(df[ft])

    # pick defined quantity TODO maybe mix elements here as they are in order of  longitude
    df = df.head(quantity_from_csv)
    X = df[features]  # set of features
    Y = df[to_estimate].values.tolist()  # continous variable

    root = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)

    root.grow_tree()

    root.print_tree()
