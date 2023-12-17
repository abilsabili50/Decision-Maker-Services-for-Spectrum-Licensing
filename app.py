from flask import Flask, request, make_response, abort
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Init
app = Flask(__name__)
scaler = MinMaxScaler()
model = load_model("model")

# gather
# df = pd.read_csv('data.csv', delimiter=",")
# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# y = df['STATUS_SIMF']
# X = df.drop('STATUS_SIMF', axis=1)
# columns = X.columns

def gather(path):
    df = pd.read_csv(path, delimiter=",")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.drop('STATUS_SIMF', axis=1)
    req_columns = df.drop(['SID_x', 'SID_y', 'SID_z'], axis=1).columns
    req_columns = req_columns.union(['SID_LAT', 'SID_LONG'])
    return df, req_columns

df, req_columns = gather('data.csv')


# preprocess
def preprocess(req_data):
    req_data['SID_x'] = np.cos(float(req_data['SID_LAT'])) * np.cos(float(req_data['SID_LONG']))
    req_data['SID_y'] = np.cos(float(req_data['SID_LAT'])) * np.sin(float(req_data['SID_LONG']))
    req_data['SID_z'] = np.sin(float(req_data['SID_LAT']))
    req_data = req_data.drop(['SID_LAT', 'SID_LONG'], axis=1)
    concated = pd.concat([df, req_data])
    encoded = pd.get_dummies(concated, columns=['SUBSERVICE'])
    req_data = encoded.iloc[[-1]]
    return scaler.fit(encoded).transform(req_data)

# ping route
@app.route('/ping')
def ping():
    return 'Successfull'

# main route (predict route)
@app.post('/predict')
def predict():
    data = {}
    for column in req_columns:
        if (column not in request.form):
            print(column)
            abort(400)
        data[column] = request.form[column]
    data = pd.DataFrame({k:[v] for k,v in data.items()})

    data = preprocess(data)

    prediction = model.predict(data)[0][0]

    return {
        "status": "OK",
        "data": {
            "STATUS_SIMF": "Prelim. Canceled" if round(prediction) >= 0.5 else "Granted",
        }
    }


if __name__ == '__main__':
    app.run(debug=True)
