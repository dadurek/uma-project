# Implementation or regression tree using roulette as test

### Authors
- Marcin Dadura
- Maciej Włodarczyk

# Read-me

This project is implementation of regression tree using roulette to choose the split. 


# Configuration

* `tree.configure(roulette_option=True)` - enable roulette, if false in function best_split() picked will be value with highest probability
* `tree.configure(rounding_amount=4)` - amount of rounding of values when printing tree
* `tree.configure(width_print=9)` - spaces padding when printing tree

# Help functions

* `dataFrame = prepare_data_frame(file_path=file_path, columns_name=np.concatenate((to_estimate, features), axis=None), size=quantity_from_csv)` - return datafram from file and parse this data
* `X, Y = prepare_data(df=dataFrame, to_estimate=to_estimate, features=features)` - return X and Y vectors needed to generate regression tree

# Usage

* `tree = Node(X=X, Y=Y, max_depth=max_depth, min_elements=min_elements)` - initialize root of tree
* `tree.grow()` - generate tree
* `tree.print_tree()` - print tree
* `tree.predict(df=dataFrame, new_column_name=new_column_name)` - predict estimated value, add new column with predicted value
* `someetig to calculate error` - 

