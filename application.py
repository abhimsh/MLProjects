from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictionDataClass

application = Flask(__name__)

app = application

@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict_page():
    if request.method == "GET":
        return render_template("prediction.html")
    elif request.method == "POST":
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")
        reading_score = int(request.form.get("reading_score"))
        writing_score = int(request.form.get("writing_score"))

        obj = PredictionDataClass()
        predicted_math_score = obj.predict(
            gender,
            race_ethnicity, 
            parental_level_of_education,
            lunch,
            test_preparation_course,
            reading_score,
            writing_score
            )
        
        return render_template("prediction.html", results=predicted_math_score[-1])

    else:
        return "UNKONOWN REQUEST RECEIVED!!!!"
    


if __name__ == "__main__":
    app.run(host="0.0.0.0")