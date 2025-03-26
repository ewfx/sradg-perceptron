from flask import Flask, render_template, request, redirect, url_for
import requests

app = Flask(__name__)

FASTAPI_URL = "http://localhost:8000"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        historical_file = request.files["historical_file"]
        current_file = request.files["current_file"]
        files = {
            "historical_file": historical_file,
            "current_file": current_file
        }
        response = requests.post(f"{FASTAPI_URL}/process", files=files)
        if response.status_code == 200:
            return redirect(url_for("result"))
        else:
            return f"Error: {response.text}", 500
    return render_template("index.html")

@app.route("/result")
def result():
    response = requests.get(f"{FASTAPI_URL}/result")
    if response.status_code == 200:
        data = response.json()
        return render_template("result.html", output_csv=data["output_csv"], actions=data["actions"])
    else:
        return f"Error: {response.text}", 404

if __name__ == "__main__":
    app.run(port=5000, debug=True)