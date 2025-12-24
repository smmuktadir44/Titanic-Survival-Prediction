import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="Template")
model = pickle.load(open("Model/Titanic_survival_Prediction.pkl","rb"))

@app.route('/')
def home():
    return render_template("titanic_prediction.html")

@app.route('predict',methods=['POST'])
def predict():
    p_class = request.form.get["pclass"]
    gender = request.form.get["sex"]
    age = int(request.form.get["age"])
    sibsph = int(request.form.get["sibsp"])
    parch = int(request.form.get["parch"])
    fare = float(request.form.get["fare"])
    port = request.form.get["embarked"]

    feature = np.array("p_class","gender","age","sibsph","parch","fare","port")
    pred = model.predict(feature)

    return render_template("titanic_prediction.html","result",pred)

if __name__=="__main__":
    app.run(debug=True)
