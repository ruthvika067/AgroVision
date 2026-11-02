import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent

DATA_FOLDER = BASE_DIR / "data"
DATA_FOLDER.mkdir(exist_ok=True)

DATA_FILE = DATA_FOLDER / "crop_yield.csv"

crops = ["Rice", "Wheat", "Maize", "Cotton", "Groundnut"]

data = []

for i in range(1000):

    crop = np.random.choice(crops)

    temperature = np.random.uniform(20, 40)
    rainfall = np.random.uniform(200, 1200)
    soil_moisture = np.random.uniform(20, 80)
    soil_ph = np.random.uniform(5, 8)

    nitrogen = np.random.uniform(20, 120)
    phosphorus = np.random.uniform(10, 80)
    potassium = np.random.uniform(10, 80)

    yield_value = (
        2
        + soil_moisture * 0.03
        + nitrogen * 0.02
        + phosphorus * 0.01
        + potassium * 0.01
        - abs(temperature - 28) * 0.05
        - abs(rainfall - 700) * 0.001
    )

    if crop == "Rice":
        yield_value += 1.5

    elif crop == "Wheat":
        yield_value += 1.0

    elif crop == "Maize":
        yield_value += 1.2

    elif crop == "Cotton":
        yield_value += 0.8

    else:
        yield_value += 0.9

    yield_value += np.random.normal(0, 0.2)

    yield_value = max(0.1, yield_value)

    data.append([
        crop,
        temperature,
        rainfall,
        soil_moisture,
        soil_ph,
        nitrogen,
        phosphorus,
        potassium,
        yield_value
    ])

columns = [
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

df = pd.DataFrame(data, columns=columns)

df.to_csv(DATA_FILE, index=False)

print("Dataset created successfully!")
print("Saved at:", DATA_FILE)