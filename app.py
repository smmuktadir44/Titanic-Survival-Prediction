import pickle
import numpy as np
from flask import Flask, request, render_template, redirect

app = Flask(__name__, template_folder="Template")

# Load model
model = pickle.load(open("Model/Titanic_survival_Prediction.pkl", "rb"))

@app.route('/')
def home():
    return render_template("titanic_prediction.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print(f"Request method: {request.method}")
    if request.method == 'GET':
        print("Redirecting GET to /")
        return redirect('/')
    try:
        pclass = int(request.form.get("pclass"))
        sex = request.form.get("sex")
        age = float(request.form.get("age"))
        sibsp = int(request.form.get("sibsp"))
        parch = int(request.form.get("parch"))
        fare = float(request.form.get("fare"))
        embarked = request.form.get("embarked")

        # Encoding (must match training)
        sex = 1 if sex == "male" else 0

        embarked_map = {"C": 0, "Q": 1, "S": 2}
        embarked = embarked_map.get(embarked, 2)

        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "SURVIVED ✅"
            result_class = "survived"
        else:
            result = "NOT SURVIVED ❌"
            result_class = "not-survived"

        return render_template(
            "titanic_prediction.html",
            prediction_text=result,
            result_class=result_class
        )

    except Exception as e:
        return render_template(
            "titanic_prediction.html",
            prediction_text=f"Error: {str(e)}",
            result_class="not-survived"
        )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
