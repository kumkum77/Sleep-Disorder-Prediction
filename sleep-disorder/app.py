from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# load model
model = joblib.load("best_model.pkl")

# load scaler
scaler = joblib.load("scaler.pkl")

# occupation mapping (same idea as LabelEncoder)
occupation_mapping = {
"Doctor":0,
"Engineer":1,
"Lawyer":2,
"Manager":3,
"Nurse":4,
"Sales Representative":5,
"Salesperson":6,
"Scientist":7,
"Software Engineer":8,
"Teacher":9
}

FEATURE_NAMES = [
'Gender','Age','Occupation','Sleep Duration','Quality of Sleep',
'Physical Activity Level','Stress Level','BMI Category',
'Heart Rate','Daily Steps','Systolic BP','Diastolic BP'
]

@app.route('/')
def home():
    return render_template("index.html", occupations=occupation_mapping.keys())


@app.route('/predict', methods=['POST'])
def predict():

    # get values from form
    gender = float(request.form['Gender'])
    age = float(request.form['Age'])

    # convert occupation text to number
    occupation_text = request.form['Occupation']
    occupation = occupation_mapping[occupation_text]

    sleep_duration = float(request.form['Sleep Duration'])
    quality_sleep = float(request.form['Quality of Sleep'])
    activity = float(request.form['Physical Activity Level'])
    stress = float(request.form['Stress Level'])
    bmi = float(request.form['BMI Category'])
    heart_rate = float(request.form['Heart Rate'])
    steps = float(request.form['Daily Steps'])
    systolic = float(request.form['Systolic BP'])
    diastolic = float(request.form['Diastolic BP'])

    # combine features
    features = np.array([[gender, age, occupation, sleep_duration,
                          quality_sleep, activity, stress, bmi,
                          heart_rate, steps, systolic, diastolic]])

    # scale features
    features_scaled = scaler.transform(features)

    # prediction
    prediction = model.predict(features_scaled)[0]

    # probability
    probabilities = model.predict_proba(features_scaled)[0]

    mapping = {
        0:"No Disorder",
        1:"Sleep Apnea",
        2:"Insomnia"
    }

    result = mapping[prediction]


    # Sleep improvement advice
    if result == "Sleep Apnea":
        advice = """
        Maintain a healthy weight and control BMI.
        Avoid alcohol before bedtime.
        Sleep on your side instead of your back.
        Maintain a regular sleep schedule.
        Consult a sleep specialist if symptoms persist.
        """

    elif result == "Insomnia":
        advice = """
        Reduce stress levels and practice relaxation techniques.
        Avoid caffeine or heavy meals before bedtime.
        Maintain consistent sleep and wake times.
        Limit mobile and screen usage before sleep.
        Try meditation, breathing exercises, or light reading.
        """

    else:
        advice = """
        Your sleep health appears normal.
        Maintain a balanced lifestyle and healthy diet.
        Continue regular physical activity.
        Maintain consistent sleep habits.
        """

    return render_template(
        "result.html",
        prediction=result,
        prob_no=round(probabilities[0]*100,2),
        prob_apnea=round(probabilities[1]*100,2),
        prob_insomnia=round(probabilities[2]*100,2),
        advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)