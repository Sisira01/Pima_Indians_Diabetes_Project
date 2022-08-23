# import relevent libararies for flask, html rendering and loading the ML model
from flask import Flask,request,url_for,redirect,render_template
import pickle
import pandas as pd
import joblib



app = Flask(__name__)

# model = pickle.load(open("model.pkl","rb"))
model = joblib.load("model.pkl")
# scale = pickle.load(open("scale.pkl","rb"))
scale = joblib.load("scale.pkl")


@app.route("/")
def landingPage():
    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def predict():
    
    pregnancies = request.form['1']
    glucose = request.form['2']
    bloodPressure = request.form['3']
    skinThickness = request.form['4']
    insulin= request.form['5']
    bmi = request.form['6']
    dpf = request.form['7']
    age = request.form['8']

    rowDF=pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])])
    rowDF_new=pd.DataFrame(scale.transform(rowDF))

    print(rowDF_new)

    #  model Prediction
    prediction = model.predict_proba(rowDF_new)
    print(f"The Predicted values is :{prediction[0][1]}")

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round val {valPred*100}%")
        return render_template('result.html',pred=f'You have chance of having diabetes.\n\n Probability of you being a diabetic is {valPred*100}%.\n\n Advice : Excersise Regularly')
    else:
        valPred =round(prediction[0][0],3)
        return render_template('result.html',pred=f'Congratulation!!!, You are in Safe Zone.\n\n Probability of you being a non-diabetic is {valPred*100:.2f}%.\n\n Advice : Excersise Regularly and maintain like this...!')
    

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True) 


