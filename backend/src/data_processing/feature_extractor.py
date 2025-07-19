# backend/src/data_processing/feature_extractor.py
import json
import base64
import numpy as np
import pandas as pd
from collections import defaultdict

def decrypt_and_decode_data(encrypted_data_b64):
    """
    Decrypts (decodes Base64) and decodes JSON from the incoming data.
    For prototype, this is just Base64 decode + JSON parse.
    """
    try:
        decoded_json_string = base64.b64decode(encrypted_data_b64).decode('utf-8')
        return json.loads(decoded_json_string)
    except Exception as e:
        print(f"Error decoding/parsing behavioral data: {e}")
        return []

def extract_web_features(events):
    """
    Extracts features from web (mouse, keyboard) events.
    Args:
        events (list): List of raw behavioral events.
    Returns:
        dict: Extracted and calculated features.
    """
    if not events:
        return {}

    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Convert to datetime objects

    features = {}

    # --- Keystroke Analysis ---
    key_down_times = {}
    flight_times = [] # Time between key up and next key down
    dwell_times = []  # Time a key is pressed

    prev_key_up_time = None
    for _, row in df.iterrows():
        if row['type'] == 'keydown':
            key_down_times[row['keyCode']] = row['timestamp']
            if prev_key_up_time:
                flight_times.append((row['timestamp'] - prev_key_up_time).total_seconds() * 1000) # ms
        elif row['type'] == 'keyup' and row['keyCode'] in key_down_times:
            dwell_time = (row['timestamp'] - key_down_times[row['keyCode']]).total_seconds() * 1000 # ms
            dwell_times.append(dwell_time)
            prev_key_up_time = row['timestamp']
            del key_down_times[row['keyCode']]

    if dwell_times:
        features['avg_dwell_time_ms'] = np.mean(dwell_times)
        features['std_dwell_time_ms'] = np.std(dwell_times)
    else:
        features['avg_dwell_time_ms'] = 0
        features['std_dwell_time_ms'] = 0

    if flight_times:
        features['avg_flight_time_ms'] = np.mean(flight_times)
        features['std_flight_time_ms'] = np.std(flight_times)
    else:
        features['avg_flight_time_ms'] = 0
        features['std_flight_time_ms'] = 0

    # Simple typing speed (chars per second approximation)
    total_time_typing = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    if total_time_typing > 0:
        features['typing_speed_cps'] = len(df[df['type'].isin(['keydown', 'keyup'])]) / 2 / total_time_typing
    else:
        features['typing_speed_cps'] = 0

    # --- Mouse Movement Analysis ---
    mouse_events = df[df['type'].isin(['mousemove', 'click'])].copy()
    if not mouse_events.empty:
        mouse_events = mouse_events.sort_values('timestamp')
        features['mouse_total_movements'] = len(mouse_events[mouse_events['type'] == 'mousemove'])
        features['mouse_total_clicks'] = len(mouse_events[mouse_events['type'] == 'click'])

        # Calculate path length and speed
        coords = mouse_events[['x', 'y']].dropna().values
        if len(coords) > 1:
            path_lengths = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            total_path_length = np.sum(path_lengths)
            total_mouse_time = (mouse_events['timestamp'].max() - mouse_events['timestamp'].min()).total_seconds()

            features['mouse_total_path_length'] = total_path_length
            if total_mouse_time > 0:
                features['mouse_avg_speed_px_per_s'] = total_path_length / total_mouse_time
            else:
                features['mouse_avg_speed_px_per_s'] = 0
        else:
            features['mouse_total_path_length'] = 0
            features['mouse_avg_speed_px_per_s'] = 0

        # Analyze mouse direction changes (basic)
        if len(coords) > 2:
            diffs = np.diff(coords, axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0]) # Angle of each segment
            angle_changes = np.abs(np.diff(angles))
            # Normalize angles to be between 0 and pi for comparison
            angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
            features['mouse_avg_angle_change_rad'] = np.mean(angle_changes)
            features['mouse_std_angle_change_rad'] = np.std(angle_changes)
        else:
            features['mouse_avg_angle_change_rad'] = 0
            features['mouse_std_angle_change_rad'] = 0

    else:
        features['mouse_total_movements'] = 0
        features['mouse_total_clicks'] = 0
        features['mouse_total_path_length'] = 0
        features['mouse_avg_speed_px_per_s'] = 0
        features['mouse_avg_angle_change_rad'] = 0
        features['mouse_std_angle_change_rad'] = 0

    # Handle cases where min/max timestamp might be the same (single event session)
    if 'timestamp' in df and len(df) > 1:
        features['session_duration_ms'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() * 1000
    else:
        features['session_duration_ms'] = 0


    # Fill any missing keys with 0 to ensure consistent feature set
    # This is important if some sessions don't have certain event types
    all_possible_features = [
        'avg_dwell_time_ms', 'std_dwell_time_ms', 'avg_flight_time_ms', 'std_flight_time_ms', 'typing_speed_cps',
        'mouse_total_movements', 'mouse_total_clicks', 'mouse_total_path_length',
        'mouse_avg_speed_px_per_s', 'mouse_avg_angle_change_rad', 'mouse_std_angle_change_rad',
        'session_duration_ms'
    ]
    for feature_name in all_possible_features:
        if feature_name not in features:
            features[feature_name] = 0.0 # Use float for consistency

    return features

def normalize_features(features, min_max_values=None):
    """
    Normalizes features using Min-Max scaling.
    In a real system, min_max_values would come from global historical data.
    For prototype, we'll use a fixed set or calculate dynamically (less ideal for real-time).
    """
    # Define a fixed set of min/max values for prototype
    # In a real system, these would be learned from a large dataset
    if min_max_values is None:
        min_max_values = {
            'avg_dwell_time_ms': {'min': 50, 'max': 500},
            'std_dwell_time_ms': {'min': 0, 'max': 200},
            'avg_flight_time_ms': {'min': 50, 'max': 1000},
            'std_flight_time_ms': {'min': 0, 'max': 300},
            'typing_speed_cps': {'min': 0, 'max': 10}, # characters per second
            'mouse_total_movements': {'min': 0, 'max': 1000},
            'mouse_total_clicks': {'min': 0, 'max': 50},
            'mouse_total_path_length': {'min': 0, 'max': 50000}, # pixels
            'mouse_avg_speed_px_per_s': {'min': 0, 'max': 1000}, # pixels per second
            'mouse_avg_angle_change_rad': {'min': 0, 'max': 3.14}, # radians (0 to pi)
            'mouse_std_angle_change_rad': {'min': 0, 'max': 1.0},
            'session_duration_ms': {'min': 0, 'max': 600000} # 10 minutes
        }

    normalized_features = {}
    for key, value in features.items():
        if key in min_max_values:
            min_val = min_max_values[key]['min']
            max_val = min_max_values[key]['max']
            if max_val - min_val > 0:
                normalized_value = (value - min_val) / (max_val - min_val)
                normalized_features[key] = max(0.0, min(1.0, normalized_value)) # Clamp to [0, 1]
            else:
                normalized_features[key] = 0.5 # Default to mid-range if no range
        else:
            normalized_features[key] = value # Keep original if no normalization info

    return normalized_features

def process_raw_data(encrypted_data_b64):
    """
    Main function to process raw behavioral data from frontend.
    """
    raw_events = decrypt_and_decode_data(encrypted_data_b64)
    if not raw_events:
        return {}
    extracted_features = extract_web_features(raw_events)
    normalized_features = normalize_features(extracted_features)
    return normalized_features

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    # Example simulated raw data (replace with actual captured data)
    # This is a minimal set. Real data would be much richer.
    example_raw_data = [
        {"type": "keydown", "keyCode": 84, "timestamp": 1678886400000},
        {"type": "keyup", "keyCode": 84, "timestamp": 1678886400080},
        {"type": "keydown", "keyCode": 69, "timestamp": 1678886400150},
        {"type": "keyup", "keyCode": 69, "timestamp": 1678886400220},
        {"type": "mousemove", "x": 100, "y": 100, "timestamp": 1678886400050},
        {"type": "mousemove", "x": 105, "y": 102, "timestamp": 1678886400100},
        {"type": "click", "x": 105, "y": 102, "timestamp": 1678886400250},
        {"type": "mousemove", "x": 200, "y": 250, "timestamp": 1678886401000},
    ]
    # Simulate Base64 encoding from frontend
    encoded_data = base64.b64encode(json.dumps(example_raw_data).encode('utf-8')).decode('utf-8')

    print("--- Testing Feature Extraction ---")
    processed_features = process_raw_data(encoded_data)
    print(json.dumps(processed_features, indent=2))