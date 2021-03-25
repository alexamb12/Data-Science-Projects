import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from find_maxleafnode import get_mae
from sklearn.tree import DecisionTreeRegressor

iowa_data = pd.read_csv('train.csv')


y = iowa_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = iowa_data[features]

# ------ define model ------
iowa_model = DecisionTreeRegressor(random_state=0)

# ------ split data ------
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# ------ fit model ------
iowa_model.fit(train_X, train_y)

# ------ find best_tree_size ------
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_node: get_mae(leaf_node, train_X, val_X, train_y, val_y) for leaf_node in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)

print(best_tree_size)

# ------ make validation predictions and calculate mean absolute error w/out specifying max_leaf_nodes------
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation Mean absolute error without specifing max_leaf nodes: {:,.0f}".format(val_mae))

# ------ calculating MAE using max_leaf_node --------
iowa_model = DecisionTreeRegressor(max_leaf_nodes=50, random_state=0)
iowa_model.fit(train_X,train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation Mean absolute error for best value of max_leaf nodes: {:,.0f}".format(val_mae))

# ------ define RandomForestRegressor model ------
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)
print("Validation Mean absolute error for RandomForest Model: {:,.0f}".format(rf_val_mae))

# ------ build RF model and train it on all X and y ------
rf_model_full_data = RandomForestRegressor(random_state=0)
rf_model_full_data.fit(X,y)

# ------ apply model to test.csv -------
test_data = pd.read_csv('test.csv')
test_X= test_data[features]
test_pred = rf_model_full_data.predict(test_X)

# ------- create .csv of output -------
output = pd.DataFrame({'Id' : test_data.Id, 
'SalePrice': test_pred})
output.to_csv('test_model.csv')
