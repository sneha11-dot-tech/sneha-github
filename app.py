import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Create the flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the features from the form
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Render the index.html template with the prediction
    return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction[0]*1000:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)