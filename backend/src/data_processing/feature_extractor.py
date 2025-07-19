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
    Extracts features from web and mobile behavioral events.
    Args:
        events (list): List of raw behavioral events.
    Returns:
        dict: Extracted and calculated features.
    """
    if not events:
        return {}

    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    features = {}

    # --- Keystroke Analysis ---
    key_down_times = {}
    flight_times = []
    dwell_times = []
    prev_key_up_time = None

    for _, row in df.iterrows():
        if row['type'] == 'keydown':
            key_down_times[row['keyCode']] = row['timestamp']
            if prev_key_up_time:
                flight_times.append((row['timestamp'] - prev_key_up_time).total_seconds() * 1000)
        elif row['type'] == 'keyup' and row['keyCode'] in key_down_times:
            dwell_time = (row['timestamp'] - key_down_times[row['keyCode']]).total_seconds() * 1000
            dwell_times.append(dwell_time)
            prev_key_up_time = row['timestamp']
            del key_down_times[row['keyCode']]

    features['avg_dwell_time_ms'] = np.mean(dwell_times) if dwell_times else 0.0
    features['std_dwell_time_ms'] = np.std(dwell_times) if dwell_times else 0.0
    features['avg_flight_time_ms'] = np.mean(flight_times) if flight_times else 0.0
    features['std_flight_time_ms'] = np.std(flight_times) if flight_times else 0.0

    total_time_typing = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    features['typing_speed_cps'] = len(df[df['type'].isin(['keydown', 'keyup'])]) / 2 / total_time_typing if total_time_typing > 0 else 0

    # --- Mouse Movement Analysis ---
    mouse_events = df[df['type'].isin(['mousemove', 'click'])].copy()
    if not mouse_events.empty:
        mouse_events = mouse_events.sort_values('timestamp')
        features['mouse_total_movements'] = len(mouse_events[mouse_events['type'] == 'mousemove'])
        features['mouse_total_clicks'] = len(mouse_events[mouse_events['type'] == 'click'])

        coords = mouse_events[['x', 'y']].dropna().values
        if len(coords) > 1:
            path_lengths = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            total_path_length = np.sum(path_lengths)
            total_mouse_time = (mouse_events['timestamp'].max() - mouse_events['timestamp'].min()).total_seconds()

            features['mouse_total_path_length'] = total_path_length
            features['mouse_avg_speed_px_per_s'] = total_path_length / total_mouse_time if total_mouse_time > 0 else 0
        else:
            features['mouse_total_path_length'] = 0
            features['mouse_avg_speed_px_per_s'] = 0

        if len(coords) > 2:
            diffs = np.diff(coords, axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_changes = np.abs(np.diff(angles))
            angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
            features['mouse_avg_angle_change_rad'] = np.mean(angle_changes)
            features['mouse_std_angle_change_rad'] = np.std(angle_changes)
        else:
            features['mouse_avg_angle_change_rad'] = 0
            features['mouse_std_angle_change_rad'] = 0
    else:
        features.update({
            'mouse_total_movements': 0,
            'mouse_total_clicks': 0,
            'mouse_total_path_length': 0,
            'mouse_avg_speed_px_per_s': 0,
            'mouse_avg_angle_change_rad': 0,
            'mouse_std_angle_change_rad': 0
        })

    # Session duration
    features['session_duration_ms'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() * 1000 if len(df) > 1 else 0

    # --- Mobile Sensor Data Analysis ---
    swipe_events = df[df['type'] == 'swipe']
    features['avg_swipe_speed'] = swipe_events['swipeSpeed'].astype(float).mean() if not swipe_events.empty else 0.0
    features['max_swipe_speed'] = swipe_events['swipeSpeed'].astype(float).max() if not swipe_events.empty else 0.0

    gyro_events = df[df['type'] == 'gyroscope']
    for axis in ['gyroX', 'gyroY', 'gyroZ']:
        if axis in gyro_events.columns:
            features[f'{axis}_stddev'] = pd.to_numeric(gyro_events[axis], errors='coerce').std()
        else:
            features[f'{axis}_stddev'] = 0.0

    accel_events = df[df['type'] == 'accelerometer']
    for axis in ['accelX', 'accelY', 'accelZ']:
        if axis in accel_events.columns:
            features[f'{axis}_mean'] = pd.to_numeric(accel_events[axis], errors='coerce').mean()
        else:
            features[f'{axis}_mean'] = 0.0

    # Ensure all features exist
    all_possible_features = [
        'avg_dwell_time_ms', 'std_dwell_time_ms', 'avg_flight_time_ms', 'std_flight_time_ms', 'typing_speed_cps',
        'mouse_total_movements', 'mouse_total_clicks', 'mouse_total_path_length',
        'mouse_avg_speed_px_per_s', 'mouse_avg_angle_change_rad', 'mouse_std_angle_change_rad',
        'session_duration_ms',
        'avg_swipe_speed', 'max_swipe_speed',
        'gyroX_stddev', 'gyroY_stddev', 'gyroZ_stddev',
        'accelX_mean', 'accelY_mean', 'accelZ_mean'
    ]
    for f in all_possible_features:
        if f not in features:
            features[f] = 0.0

    return features


def normalize_features(features, min_max_values=None):
    if min_max_values is None:
        min_max_values = {
            'avg_dwell_time_ms': {'min': 50, 'max': 500},
            'std_dwell_time_ms': {'min': 0, 'max': 200},
            'avg_flight_time_ms': {'min': 50, 'max': 1000},
            'std_flight_time_ms': {'min': 0, 'max': 300},
            'typing_speed_cps': {'min': 0, 'max': 10},
            'mouse_total_movements': {'min': 0, 'max': 1000},
            'mouse_total_clicks': {'min': 0, 'max': 50},
            'mouse_total_path_length': {'min': 0, 'max': 50000},
            'mouse_avg_speed_px_per_s': {'min': 0, 'max': 1000},
            'mouse_avg_angle_change_rad': {'min': 0, 'max': 3.14},
            'mouse_std_angle_change_rad': {'min': 0, 'max': 1.0},
            'session_duration_ms': {'min': 0, 'max': 600000},
            'avg_swipe_speed': {'min': 0, 'max': 2000},
            'max_swipe_speed': {'min': 0, 'max': 3000},
            'gyroX_stddev': {'min': 0, 'max': 1.0},
            'gyroY_stddev': {'min': 0, 'max': 1.0},
            'gyroZ_stddev': {'min': 0, 'max': 1.0},
            'accelX_mean': {'min': -1.0, 'max': 1.0},
            'accelY_mean': {'min': -1.0, 'max': 1.0},
            'accelZ_mean': {'min': 0.5, 'max': 2.0}
        }

    normalized = {}
    for k, v in features.items():
        if k in min_max_values:
            min_val = min_max_values[k]['min']
            max_val = min_max_values[k]['max']
            if max_val - min_val > 0:
                normalized[k] = max(0.0, min(1.0, (v - min_val) / (max_val - min_val)))
            else:
                normalized[k] = 0.5
        else:
            normalized[k] = v
    return normalized

def process_raw_data(encrypted_data_b64):
    raw_events = decrypt_and_decode_data(encrypted_data_b64)
    if not raw_events:
        return {}
    extracted = extract_web_features(raw_events)
    normalized = normalize_features(extracted)
    return normalized
