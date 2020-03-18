
# API created of ML Model using python Flask Library
# Post method used to perform the prediction using loaded model in pickle

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Load the Model

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')

def home():
 return render_template('index.html')

@app.route('/api',methods=['POST'])
# Features have been loaded from model and called from index.html
def predict():

 features = [int(x) for x in request.form.values()]
 prediction_features = [np.array(features)]
 prediction = model.predict(prediction_features)

 output = round(prediction[0], 2)

 return render_template('index.html', prediction_text='Annual Income should be $ {}'.format(output))

#Run the server

if __name__ == "__main__":

 app.run(debug=True)


# Once run the app.py it will connect to http://127.0.0.1:5000/  server , we can open this link and chceck the model prediction data
