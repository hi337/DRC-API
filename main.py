import flask
from flask import request, jsonify
import tensorflow as tf
from datetime import datetime, timedelta, date
import requests

app = flask.Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("83%.h5")

props_arr = [
    "temperature_2m",
    "relativehumidity_2m",
    "precipitation",
    "windspeed_10m",
    "soil_moisture_0_to_7cm",
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.method == "POST":
            data = request.get_json(force=True)

            # Extract the 5 features (temperature, humidity, precipitation, windspeed, moisture)
            longitude = data["longitude"]
            latitude = data["latitude"]

            today = date.today()

        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={today}&end_date={today}&hourly=temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,soil_moisture_0_to_7cm"

        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request was not successful

        # Parse the JSON response
        data = response.json()
        return_array = []

        for item in props_arr:
            vals = data["hourly"][item]
            if len(vals) > 0:
                avg = sum(vals) / len(vals)
                return_array.append(avg)

            # Prepare the input data for prediction
            input_data = [[temperature, humidity, precipitation, windspeed, moisture]]

            # Make a prediction
            prediction = model.predict(input_data)

            # Create a response JSON
            response = {"prediction": float(prediction[0][0])}

            return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
