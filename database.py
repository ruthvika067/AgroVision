import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "agrovision.db"


def create_table():
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop TEXT,
                temperature REAL,
                rainfall REAL,
                soil_moisture REAL,
                soil_ph REAL,
                nitrogen REAL,
                phosphorus REAL,
                potassium REAL,
                predicted_yield REAL
            )
        """)


def save_prediction(data, predicted_yield):
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("""
            INSERT INTO predictions (
                crop,
                temperature,
                rainfall,
                soil_moisture,
                soil_ph,
                nitrogen,
                phosphorus,
                potassium,
                predicted_yield
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["crop"],
            data["temperature"],
            data["rainfall"],
            data["soil_moisture"],
            data["soil_ph"],
            data["nitrogen"],
            data["phosphorus"],
            data["potassium"],
            predicted_yield
        ))


def get_history():
    with sqlite3.connect(DATABASE_FILE) as connection:
        cursor = connection.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
        """)
        return cursor.fetchall()