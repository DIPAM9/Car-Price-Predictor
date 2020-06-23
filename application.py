from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        company = request.form['company']
        year = request.form['year']
        kms_driven = request.form['kms_driven']
        fuel_type = request.form['fuel_type']
        result = predictor(company, year, kms_driven, fuel_type)
    return render_template('index1.html', result=result)


def predictor(company, year, kms_driven, fuel_type):
    data = pd.read_csv('quikr_car.csv')
    data['Price'] = data['Price'].str.replace(',', '')
    data['Price'] = data['Price'].replace('Ask For Price',
                                          data[data['Price'] != 'Ask For Price']['Price'].astype(float).mean())
    data['Price'] = data['Price'].astype(float)
    data['kms_driven'] = data['kms_driven'].str.replace('kms', '')
    data['kms_driven'] = data['kms_driven'].str.replace(',', '')
    data = data.drop(890)
    data = data.drop(891)
    data['kms_driven'] = data['kms_driven'].astype(float)
    data['kms_driven'].fillna(data['kms_driven'].mean(), inplace=True)
    data['year'][data['year'].str.startswith('20') == False] = np.nan
    data['year'].fillna(data['year'].astype(float).mean(), inplace=True)
    data['year'] = data['year'].astype(float)
    data.fillna({'fuel_type': 'another'}, inplace=True)
    data['fuel_type'].replace({'Petrol': 0, 'Diesel': 1, 'LPG': 2, 'another': 3}, inplace=True)
    data['fuel_type'] = data['fuel_type'].astype(float)
    mask = data['company'].value_counts()
    p = mask[mask >= 5].index
    data['company'] = np.where(data['company'].isin(p), data['company'], 'Other')
    data_train = pd.get_dummies(data, columns=['company'])
    z = sorted(data['company'].value_counts().index)
    z.insert(0, 'name')
    z.insert(1, 'year')
    z.insert(2, 'Price')
    z.insert(3, 'kms_driven')
    z.insert(4, 'fuel_type')
    data_train.columns = z
    X = data_train.drop(columns=['name', 'Price']).iloc[:, :].values
    y = data_train.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    r = Ridge(alpha=0.0001)
    r.fit(X_train, y_train)
    arr = np.array([company, year, kms_driven, fuel_type])
    if arr[3] == 'Petrol':
        arr[3] = 0
    elif arr[3] == 'Diesel':
        arr[3] = 1
    elif arr[3] == 'LPG':
        arr[3] = 2
    else:
        arr[3] = 3
    for i in range(5, len(data_train.columns)):
        if data_train.columns[i] == company:
            arr = np.append(arr, 1)
        else:
            arr = np.append(arr, 0)
    arr = np.delete(arr, 0)
    arr = arr.astype(float)
    arr = arr.reshape(-1, len(arr))
    return r.predict(arr)[0]


if __name__ == '__main__':
    app.run(debug=True)
