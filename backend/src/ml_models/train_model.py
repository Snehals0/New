import os
import json
import joblib
import numpy as np
from pymongo import MongoClient
from sklearn.ensemble import IsolationForest
import sys
sys.path.append(os.path.abspath("src"))

from data_processing.feature_extractor import normalize_features

# --- Config ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "behavioral_logs"
COLLECTION_NAME = "session_logs_collection"
MODEL_OUTPUT_PATH = "backend/src/ml_models/model.joblib"

# --- Connect to MongoDB ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- Define feature order ---
FEATURE_ORDER = [
    'avg_dwell_time_ms', 'std_dwell_time_ms', 'avg_flight_time_ms', 'std_flight_time_ms', 'typing_speed_cps',
    'mouse_total_movements', 'mouse_total_clicks', 'mouse_total_path_length',
    'mouse_avg_speed_px_per_s', 'mouse_avg_angle_change_rad', 'mouse_std_angle_change_rad',
    'session_duration_ms',
    'avg_swipe_speed', 'max_swipe_speed',
    'gyroX_stddev', 'gyroY_stddev', 'gyroZ_stddev',
    'accelX_mean', 'accelY_mean', 'accelZ_mean'
]

# --- Load session data ---
print("Fetching session features from MongoDB...")
sessions = collection.find({"processed_features": {"$exists": True}})
data = []

for session in sessions:
    raw_features = session.get("processed_features", {})
    normalized = normalize_features(raw_features)

    # Ensure consistent order
    feature_vector = [normalized.get(f, 0.0) for f in FEATURE_ORDER]
    data.append(feature_vector)

print(f"Total sessions loaded: {len(data)}")

if len(data) < 10:
    print("❌ Not enough data to train the model. Need at least 10 sessions.")
    exit(1)

# --- Train IsolationForest model ---
print("Training IsolationForest...")
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(np.array(data))

# --- Save the model ---
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)

print(f"✅ Model trained and saved to {MODEL_OUTPUT_PATH}")
