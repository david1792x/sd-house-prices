## **Modelling our data with machine learning**

Once our data is cleaned, we can move over to preprocessing it and creating a model for predictions. We are looking to build a supervised machine learning model that solves a regression task by inputting tabular data.

To start off, we import the required modules and create a preprocessing pipeline that takes our clean dataset and transforms it to a model-friendly representation. To do this, we separate our data into numerical and categorical features, which will be treated differently by our preprocessing pipeline with the help of a scikit-learn ColumnTransformer. The numerical features will be passed through unchanged, since we will use a tree-based model (XGBoost) that does not require feature scaling (standarization), and the only thing we need to do is encode our categorical data using One-Hot encoding. We encode the home type and city variables to finish our preprocessor transformer.


```python
# Import required modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config
set_config(transform_output = "pandas")

# Create preprocessing pipeline
numeric_features = ['bedrooms', 'bathrooms', 'livingArea', 'longitude', 'latitude']
categorical_features = ['homeType', 'city']
preprocessor = ColumnTransformer([('ohe', OneHotEncoder(sparse_output= False), categorical_features),
                                  ('passthrough', 'passthrough' , numeric_features)],
                                remainder = 'drop')
```

Now, we can move over to making our final pipeline, which connects our preprocessor to an estimator that will learn the underlying patterns in the data and output a price prediction. We will use an extreme gradient boosting tree model, since it is one of the best performing models when working with tabular data, whose hyperparameters have been tuned using cross-validation.


```python
# Import ML libraries
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create pipeline of preprocessing and model (XGBoost with tuned hyperparameters)
pipeline = Pipeline([('preprocessor', preprocessor), ('estimator', XGBRegressor(learning_rate = 0.1, max_depth = 7, n_estimators = 200, reg_lambda = 0.1, reg_alpha = 0.1))])
```

Now that we have defined the complete model pipeline, we can train and test the model using cross validation and assess the quality of the model's performance. We will use the mean absolute error, root mean square error, and R² value to rate our model's performance.


```python
# Split data into features and target
X = df.drop('price', axis=1)
y = df['price']

# Test model performance with 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=36)
mae = []
rmse = []
r2 = []
average_error = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae.append(mean_absolute_error(y_test, y_pred))
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2.append(r2_score(y_test, y_pred))
    average_error.append(np.mean(np.abs((y_test - y_pred) / y_test)))

print(f'MAE: ${np.mean(mae):.0f}')
print(f'RMSE: ${np.mean(rmse):.0f}')
print(f'R2 score: {np.mean(r2):.4f} or {np.mean(r2)*100:.2f}%')
print(f'Average error: {np.mean(average_error):.4f} or {np.mean(average_error)*100:.2f}%')
print(f'Average error standard deviation: {np.std(average_error):.4f} or {np.std(average_error)*100:.2f}%')
```

    MAE: $146226
    RMSE: $248083
    R2 score: 0.8634 or 86.34%
    Average error: 0.1412 or 14.12%
    Average error standard deviation: 0.0025 or 0.25%
    

The mean absolute error (MAE) shows the mean difference between actual and predicted price, while the root mean squared error (RMSE) shows the square root of the differences of the squared values (penalizes large differences). We can see that the mean of the cross validated errors are \$146,226 for the mean absolute error, and \$248,083 for the mean squared error. This errors are calculated across all price ranges, so they are most notable in high value homes, as we will see further on. The R² value is one of the most important metrics for quantifying our model performance, and it describes how much of the variance in the price can be attributed to the predictor variables that our model used. In this case, R² is 0.8634, which tells us that our model is capable of explaining 86.34% of the variance in house price, which is pretty high. The average error determines how far the prediction is from the actual price across all listings, and in this case the mean error is 14.12% with a standard deviation of 0.25% (13.87% to 14.37%).

Now, we will split our data into a training (80%) and testing (20%) set, train the model on the training set, and predict the prices on the testing set. As we can see, the predicted price is pretty close to the actual price on most examples, and while there is a variation, the error is relatively small in percentage across price ranges.


```python
# Predict on test set and compare to actual values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).astype(int)
predictions['Error'] = np.abs(predictions['Actual'] - predictions['Predicted'])
predictions['Error (%)'] = np.abs((predictions['Actual'] - predictions['Predicted']) / predictions['Actual'])*100
predictions['Actual'] = predictions['Actual'].map('${:.0f}'.format)
predictions['Predicted'] = predictions['Predicted'].map('${:.0f}'.format)
predictions['Error'] = predictions['Error'].map('${:.0f}'.format)
predictions['Error (%)'] = predictions['Error (%)'].map('{:.2f}%'.format)
predictions.head(10)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
      <th>Error</th>
      <th>Error (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6577</th>
      <td>$1585000</td>
      <td>$1686247</td>
      <td>$101247</td>
      <td>6.39%</td>
    </tr>
    <tr>
      <th>14922</th>
      <td>$1559000</td>
      <td>$1532907</td>
      <td>$26093</td>
      <td>1.67%</td>
    </tr>
    <tr>
      <th>20201</th>
      <td>$800000</td>
      <td>$713347</td>
      <td>$86653</td>
      <td>10.83%</td>
    </tr>
    <tr>
      <th>15134</th>
      <td>$795000</td>
      <td>$921897</td>
      <td>$126897</td>
      <td>15.96%</td>
    </tr>
    <tr>
      <th>3869</th>
      <td>$810000</td>
      <td>$837964</td>
      <td>$27964</td>
      <td>3.45%</td>
    </tr>
    <tr>
      <th>863</th>
      <td>$1910000</td>
      <td>$1735608</td>
      <td>$174392</td>
      <td>9.13%</td>
    </tr>
    <tr>
      <th>3909</th>
      <td>$800000</td>
      <td>$708037</td>
      <td>$91963</td>
      <td>11.50%</td>
    </tr>
    <tr>
      <th>13075</th>
      <td>$755000</td>
      <td>$742952</td>
      <td>$12048</td>
      <td>1.60%</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>$1770000</td>
      <td>$1662835</td>
      <td>$107165</td>
      <td>6.05%</td>
    </tr>
    <tr>
      <th>20472</th>
      <td>$525000</td>
      <td>$547530</td>
      <td>$22530</td>
      <td>4.29%</td>
    </tr>
  </tbody>
</table>
</div>


Something interesting that we can extract from our model, since it is tree-based, is the importance of the features in determining a prediction. The most important factor in determining house price is whether the house is a mobile home (manufactured) or not. After that, the living area is the second most important factor, and after those two, the city or location of the house is the next factor to take into account. Some areas like Coronado, Clairemont and La Jolla, among others, have a clearer price range than other areas.


```python
# Get feature importances from model
feature_importances = pd.DataFrame(pipeline['estimator'].feature_importances_, index = pipeline['preprocessor'].get_feature_names_out(), columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head(10)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ohe__homeType_MANUFACTURED</th>
      <td>0.261593</td>
    </tr>
    <tr>
      <th>passthrough__livingArea</th>
      <td>0.110888</td>
    </tr>
    <tr>
      <th>ohe__city_Coronado</th>
      <td>0.100577</td>
    </tr>
    <tr>
      <th>ohe__city_Clairemont</th>
      <td>0.076101</td>
    </tr>
    <tr>
      <th>ohe__city_Carlsbad</th>
      <td>0.061977</td>
    </tr>
    <tr>
      <th>ohe__city_Poway</th>
      <td>0.046483</td>
    </tr>
    <tr>
      <th>passthrough__longitude</th>
      <td>0.036927</td>
    </tr>
    <tr>
      <th>passthrough__latitude</th>
      <td>0.035856</td>
    </tr>
    <tr>
      <th>ohe__city_Rancho Santa Fe</th>
      <td>0.035648</td>
    </tr>
    <tr>
      <th>ohe__city_La Jolla</th>
      <td>0.034348</td>
    </tr>
  </tbody>
</table>
</div>


Finally, we can use our model to predict house prices of synthetic, or made up data. We define three different homes, two regular single family homes, and one mobile home, located in Chula Vista, La Jolla and Poway respectively. The living area is chosen according to the selected number of bedrooms and bathrooms and the coordinates are taken from the mean values of the chosen zone. These three homes describe homes at an expected medium, high and low price ranges respectively


```python
# Predict house prices on synthetic data
synthetic_data = pd.DataFrame({'bedrooms': [3, 5, 2],
                                'bathrooms': [2, 3, 2],
                                'livingArea': [1750, 3500, 1500],
                                'homeType': ['SINGLE_FAMILY', 'SINGLE_FAMILY', 'MANUFACTURED'],
                                'city': ['Chula Vista', 'La Jolla', 'Poway'],
                                'longitude': [-117.0032, -117.2549, -117.0407],
                                'latitude': [32.6277, 32.840, 32.9766]})

synthetic_data['predictedPrice'] = pipeline.predict(synthetic_data).astype(int)
synthetic_data
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>livingArea</th>
      <th>homeType</th>
      <th>city</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>predictedPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>2</td>
      <td>1750</td>
      <td>SINGLE_FAMILY</td>
      <td>Chula Vista</td>
      <td>-117.0032</td>
      <td>32.6277</td>
      <td>799958</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>3</td>
      <td>3500</td>
      <td>SINGLE_FAMILY</td>
      <td>La Jolla</td>
      <td>-117.2549</td>
      <td>32.8400</td>
      <td>3789290</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>1500</td>
      <td>MANUFACTURED</td>
      <td>Poway</td>
      <td>-117.0407</td>
      <td>32.9766</td>
      <td>305865</td>
    </tr>
  </tbody>
</table>
</div>


The model predicted a price of \$799,958 for the medium priced home, \$3,789,290 for the expensive home, and \$305,865 for the cheap home. These results are comparable to the actual house prices of real, similar home. The final step is to save our model in order to create a simple web app that allows a user to input the required data, and returns a price estimate for that home.


```python
# Save model as pickle file
import pickle
file_name = 'sd_pipeline.pkl'
pickle.dump(pipeline, open(file_name, 'wb'))
```

Now, we can open our model using Streamlit and make predictions with the saved pipeline using data with the same format as the one used to train it.
