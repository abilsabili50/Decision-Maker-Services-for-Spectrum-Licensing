from flask import Flask, request, make_response, abort
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import load_model
import pandas as pd

# Init
app = Flask(__name__)
scaler = MinMaxScaler()
model = load_model("model")

# gather
df = pd.read_csv('data.csv', delimiter=",")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
y = df['STATUS_SIMF']
X = df.drop('STATUS_SIMF', axis=1)
columns = X.columns

# preprocess
def preprocess(req_data):
    concated = pd.concat([X, req_data])
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
    for field in columns:
        if (field not in request.form):
            print(field)
            abort(400)
        data[field] = request.form[field]
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
