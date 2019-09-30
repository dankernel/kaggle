
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Read CSV
file_path = "./train.csv"
data = pd.read_csv(file_path)

# Set X, y
# features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]
y = data.SalePrice

def get_MAE_DTR(max_leaf_nodes, train_X, val_X, train_y, val_y):

    # DecisionTreeRegressor Model
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)

    # Predict
    predictions = model.predict(val_X)
    MAE = mean_absolute_error(val_y, predictions)

    return MAE

def get_MAE_RFR(train_X, val_X, train_y, val_y):

    # RFR model
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)

    # Predict
    predictions = model.predict(val_X)
    MAE = mean_absolute_error(val_y, predictions)

    return MAE
    
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

print('RFR', get_MAE_RFR(train_X, val_X, train_y, val_y))
