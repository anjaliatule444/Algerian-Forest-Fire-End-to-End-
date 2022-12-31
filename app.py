import pickle
from flask import app,request,Flask,Request,jsonify,url_for,render_template
import pandas as pd


app=Flask(__name__)
Classification_model=pickle.load(open("Decision_Tree_Classifier.pkl","rb"))
Regression_model=pickle.load(open("random_forest_regressor.pkl","rb"))

#@app.route('/')
#def home():
#    return render_template('home.html')

## Postman-Single Input-Classification
@app.route("/predict_api_classification_si",methods=['POST'])
def predict_api_classification_si():
    data=request.json['single_input']
    new_2D_data=[(list(data.values()))]
    output = Classification_model.predict(new_2D_data)[0]
    if(output==1):
        return "Not Fire"
    else:
        return "Fire"

## Postman-Single Input-Regression
@app.route("/predict_api_regression_si",methods=['POST'])
def predict_api_regression_si():
    data=request.json['single_input_reg']
    new_2D_data=[(list(data.values()))]
    output = Regression_model.predict(new_2D_data)[0]
    return str(output)

## Postman-Batch Input-Classification
@app.route("/predict_api_classification_bi",methods=['POST'])
def predict_api_classification_bi():
    data=request.json['batch_input_classification']
    new_data=pd.DataFrame(data)
    output = Classification_model.predict(new_data)
    output1=[]
    for i in output:
        if(i==1):
            output1.append("Not Fire")
        else:
            output1.append("Fire")
    return str(output1)

## Postman-Batch Input-Regression
@app.route("/predict_api_regression_bi",methods=['POST'])
def predict_api_regression_bi():
    data=request.json['batch_input_reg']
    new_data=pd.DataFrame(data)
    output = Regression_model.predict(new_data)
    return str(output)


if __name__=="__main__":
    app.run(debug=True)