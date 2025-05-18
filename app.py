
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar modelo
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def formulario():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form.get(var)) for var in request.form
            ]
            prediction = model.predict([features])[0]
        except Exception as e:
            prediction = f"Error en predicción: {str(e)}"
    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
