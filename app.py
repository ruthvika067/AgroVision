from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

from database import create_table, save_prediction, get_history

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"]
model_columns = saved_model["columns"]

ALLOWED_CROPS = [
    "Rice",
    "Wheat",
    "Maize",
    "Cotton",
    "Groundnut"
]

create_table()


def predict_yield(data):
    input_data = pd.DataFrame([data])

    input_data = pd.get_dummies(
        input_data,
        columns=["crop"]
    )

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    result = model.predict(input_data)[0]
    return max(0, round(float(result), 2))


def get_recommendations(data):
    recommendations = []

    if data["soil_ph"] < 5.5:
        recommendations.append(
            "Your entered soil pH is acidic. Consider a soil test "
            "and consult a local agricultural officer before treatment."
        )
    elif data["soil_ph"] > 7.5:
        recommendations.append(
            "Your entered soil pH is alkaline. Check soil-test "
            "recommendations before applying amendments."
        )
    else:
        recommendations.append(
            "Your entered soil pH is within a commonly suitable range "
            "for many crops. Crop-specific needs may differ."
        )

    if data["soil_moisture"] < 30:
        recommendations.append(
            "Soil moisture is entered as low. Check the crop's water "
            "needs and local conditions before irrigation."
        )
    elif data["soil_moisture"] > 70:
        recommendations.append(
            "Soil moisture is entered as high. Check drainage and "
            "avoid unnecessary irrigation."
        )
    else:
        recommendations.append(
            "Review soil moisture regularly, especially during "
            "important crop growth stages."
        )

    if data["rainfall"] < 300:
        recommendations.append(
            "Entered rainfall is relatively low. Plan water use "
            "according to crop stage and local advice."
        )
    else:
        recommendations.append(
            "Track rainfall during the season and adjust irrigation "
            "based on actual field conditions."
        )

    recommendations.append(
        "Use a recent soil test to guide nitrogen, phosphorus, "
        "and potassium application."
    )

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    recommendations = []
    error = None

    if request.method == "POST":
        try:
            crop = request.form.get("crop", "").strip()
            location = request.form.get("location", "").strip()
            land_area = float(request.form.get("land_area", ""))

            data = {
                "crop": crop,
                "temperature": float(request.form.get("temperature", "")),
                "rainfall": float(request.form.get("rainfall", "")),
                "soil_moisture": float(request.form.get("soil_moisture", "")),
                "soil_ph": float(request.form.get("soil_ph", "")),
                "nitrogen": float(request.form.get("nitrogen", "")),
                "phosphorus": float(request.form.get("phosphorus", "")),
                "potassium": float(request.form.get("potassium", ""))
            }

            if crop not in ALLOWED_CROPS:
                raise ValueError("Please select a valid crop.")

            if not location:
                raise ValueError("Enter your village or district.")

            if land_area <= 0:
                raise ValueError("Farm size must be greater than zero.")

            if not 0 <= data["soil_moisture"] <= 100:
                raise ValueError("Soil moisture must be between 0 and 100.")

            if not 0 <= data["soil_ph"] <= 14:
                raise ValueError("Soil pH must be between 0 and 14.")

            if data["rainfall"] < 0:
                raise ValueError("Rainfall cannot be negative.")

            if any(
                data[field] < 0
                for field in ["nitrogen", "phosphorus", "potassium"]
            ):
                raise ValueError("NPK values cannot be negative.")

            yield_per_acre = predict_yield(data)

            total_yield = round(
                yield_per_acre * land_area,
                2
            )

            prediction = {
                **data,
                "location": location,
                "land_area": land_area,
                "yield_per_acre": yield_per_acre,
                "total_yield": total_yield
            }

            recommendations = get_recommendations(data)

            save_prediction(data, yield_per_acre)

        except (ValueError, TypeError) as error_message:
            error = str(error_message)

    history = get_history()

    return render_template(
        "index.html",
        prediction=prediction,
        recommendations=recommendations,
        history=history,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)