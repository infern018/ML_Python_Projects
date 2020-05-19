import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

home_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
#target
y = home_data.SalePrice
#features to be taken:
features  = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'YearBuilt', 'BedroomAbvGr', 'GrLivArea']

X = home_data[features]

from sklearn.ensemble import RandomForestRegressor

#model:
rf_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

rf_model.fit(X, y)

test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

test_X = test_data[features]

#taking care if empty values:
test_X = test_X.fillna(test_X.mean())

preds = rf_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': preds})
output.to_csv('submission_1.csv', index=False)