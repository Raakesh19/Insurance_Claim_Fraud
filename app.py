import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
model = pickle.load(open('dt_model.pkl','rb'))
scaler= pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        
        Months_As_Customer = int(request.form['months_as_customer'])
        Age = int(request.form['age'])
        Umbrella_limit = int(request.form['umbrella_limit'])
        Incident_severity = (request.form['incident_severity'])
        if Incident_severity =='Trivial Damage':
            Incident_severity = 1
        elif Incident_severity =='Minor Damage':
            Incident_severity = 2
        elif Incident_severity =='Major Damage':
            Incident_severity =3
        else: Incident_severity =4

        Bodily_Injuries = int(request.form['bodily_injuries'])
        Police_Report_Available =(request.form['police_report_avaliable'])
        if Police_Report_Available =='YES':
            Police_Report_Available = 1
        else:
            Police_Report_Available = 0    
        Total_Claim_Amount =int(request.form['total_claim_amount'])
        Injury_Claim =int(request.form['injury_claim'])
        Vehicle_Claim =int(request.form['vehicle_claim'])
        
        final_input= scaler.fit_transform(np.array([Months_As_Customer,Age,Umbrella_limit, Incident_severity,
        Bodily_Injuries,Police_Report_Available, Total_Claim_Amount, Injury_Claim, Vehicle_Claim]).reshape(1,-1))
        
        output = model.predict(final_input)[0]
        if output == 0:
            return(render_template("index.html", prediction_text="The insurance claim is not Fraudulent."))
        else:
            return render_template("index.html", prediction_text="The insurance claim is Fraudulent.")
    else:
        return render_template('index.html')        
 
 
if __name__ =="__main__":
    app.run(debug=True)       




    # data=[float(x) for x in request.form.values()]
    # final_input=scaler.transform(np.array(data).reshape(1,-1))
    # print(final_input)
    # output=dt_model.predict(final_input)[0]
    # if output ==1:
    #     return(render_template("index.html",prediction_text="The insurance claim is Fraudulent."))
    # else:
    #     return render_template("index.html",prediction_text="The insurance claim is not Fraudulent.")



