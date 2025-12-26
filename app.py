from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


#load the trained machine learning model
model = joblib.load('house_price_model.pkl')



@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # get input values from the form
            feature1 = float(request.form['feature1']) # OverallQual
            feature2 = float(request.form['feature2']) # GrLivArea
            feature3 = float(request.form['feature3']) # GarageCars
            feature4 = float(request.form['feature4']) # TotalBsmtSF

            input_data = np.array([[feature1, feature2, feature3, feature4]])

            #predict
            result = model.predict(input_data)[0]
            prediction = round(result, 2)
    
        except Exception as e:
            prediction = f'Error: {e}'

    return render_template("index.html", prediction = prediction)



if __name__ == '__main__':
    app.run(debug=True)
