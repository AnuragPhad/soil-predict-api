from flask import Flask, request, jsonify
import joblib  # or keras.models.load_model for .h5
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load('model.pkl')  # Or use keras.models.load_model('model.h5')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)