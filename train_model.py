import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "crop_yield.csv"

MODEL_FOLDER = BASE_DIR / "model"
MODEL_FOLDER.mkdir(exist_ok=True)

MODEL_FILE = MODEL_FOLDER / "crop_model.pkl"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Run generate_dataset.py first."
    )

df = pd.read_csv(DATA_FILE)

required_columns = [
    "crop",
    "temperature",
    "rainfall",
    "soil_moisture",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "yield"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Dataset is missing these columns: "
        + ", ".join(missing_columns)
    )

df = df[required_columns].dropna()

X = df.drop(columns=["yield"])
y = df["yield"]

X = pd.get_dummies(X, columns=["crop"])
model_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

joblib.dump(
    {
        "model": model,
        "columns": model_columns
    },
    MODEL_FILE
)

print("Model trained successfully.")
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 3))
print("Model saved to:", MODEL_FILE)