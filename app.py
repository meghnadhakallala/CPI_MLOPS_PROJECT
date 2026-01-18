from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":
        cpi_values = request.form["cpi"].split(",")
        cpi_values = [float(x.strip()) for x in cpi_values]

        response = requests.post(API_URL, json=cpi_values)
        prediction = response.json()

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
