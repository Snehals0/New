# backend/src/ml_models/anomaly_detector.py
import numpy as np
from sklearn.ensemble import IsolationForest # A simple anomaly detection model for demo
import joblib # To save/load models

# In a real application, you would load a pre-trained model here.
# For the prototype, we'll simulate a simple detection logic or train a dummy model.

# Define feature order for consistency (must match order from feature_extractor)
FEATURE_ORDER = [
    'avg_dwell_time_ms', 'std_dwell_time_ms', 'avg_flight_time_ms', 'std_flight_time_ms', 'typing_speed_cps',
    'mouse_total_movements', 'mouse_total_clicks', 'mouse_total_path_length',
    'mouse_avg_speed_px_per_s', 'mouse_avg_angle_change_rad', 'mouse_std_angle_change_rad',
    'session_duration_ms'
]

# --- Simple Rule-Based Anomaly Detection (for initial prototype) ---
def calculate_rule_based_risk_score(current_features, user_profile):
    """
    Calculates a risk score based on deviations from the user's profile.
    Higher deviation = higher risk.
    """
    risk_score = 0.0
    deviation_sum = 0.0
    feature_count = 0

    # Define sensitivity for each feature (how much deviation contributes to risk)
    # These are arbitrary for prototype; would be tuned in real system
    sensitivity = {
        'avg_dwell_time_ms': 0.2,
        'std_dwell_time_ms': 0.1,
        'avg_flight_time_ms': 0.2,
        'std_flight_time_ms': 0.1,
        'typing_speed_cps': 0.3,
        'mouse_avg_speed_px_per_s': 0.2,
        'mouse_total_clicks': 0.1,
        'mouse_total_path_length': 0.1,
        'mouse_avg_angle_change_rad': 0.1,
        'mouse_total_movements': 0.1,
        'session_duration_ms': 0.05 # Less sensitive for duration
    }

    for feature_name in FEATURE_ORDER:
        current_val = current_features.get(feature_name, 0.0)
        profile_val = user_profile.get(feature_name, 0.0)

        # Avoid division by zero for profile_val if it's 0
        if profile_val == 0.0 and current_val == 0.0:
            deviation = 0.0
        elif profile_val == 0.0: # If profile is zero but current is not, it's a significant deviation
            deviation = 1.0 # Max deviation
        else:
            deviation = abs(current_val - profile_val) / profile_val

        # Apply sensitivity and cap deviation for a single feature
        deviation_sum += min(1.0, deviation) * sensitivity.get(feature_name, 0.1) # Max 1.0 per feature

        feature_count += 1

    # Simple average deviation, scaled to be between 0 and 1
    if feature_count > 0:
        # Normalize total deviation sum by the total sensitivity to get a score 0-1
        total_sensitivity = sum(sensitivity.values())
        risk_score = min(1.0, deviation_sum / total_sensitivity)
    else:
        risk_score = 0.0

    # Add a baseline risk if profile is new/empty (e.g., 0.2 for unknown behavior)
    if not any(user_profile.values()): # If all profile features are 0 or None
        risk_score = max(risk_score, 0.2) # Ensure some risk for unprofiled users

    return risk_score


# --- Placeholder for ML Model-Based Anomaly Detection ---
# For a full implementation, you'd train a model (e.g., IsolationForest, Autoencoder, LSTM)
# using historical data and save it. Here's a conceptual placeholder.

_model = None # Placeholder for a loaded ML model

def _load_model():
    """
    Loads a pre-trained ML model. For prototype, we'll simulate a dummy model.
    In a real scenario, this would load a file from disk.
    """
    global _model
    if _model is None:
        # Create a dummy IsolationForest model for illustration
        # In real use, this would be trained on real data
        _model = IsolationForest(contamination=0.05, random_state=42)
        # Simulate training with some random data (DO NOT USE IN PROD)
        dummy_data = np.random.rand(100, len(FEATURE_ORDER))
        _model.fit(dummy_data)
        print("Dummy ML model (IsolationForest) initialized.")
    return _model

def calculate_ml_based_risk_score(current_features_dict):
    """
    Calculates a risk score using a pre-trained ML model.
    Returns a score between 0 (normal) and 1 (highly anomalous).
    """
    model = _load_model()
    if model is None:
        print("ML model not loaded, returning default risk.")
        return 0.5 # Default to moderate risk if model isn't ready

    # Convert dictionary of features to a numpy array in consistent order
    features_array = np.array([current_features_dict.get(f, 0.0) for f in FEATURE_ORDER]).reshape(1, -1)

    # IsolationForest score_samples returns the anomaly score.
    # Lower score means more anomalous (closer to -1 or -0.5 typical values).
    # We need to invert and normalize it to a 0-1 risk score.
    # Typical IsolationForest scores range from -0.5 (very anomalous) to 0.5 (normal)
    raw_score = model.decision_function(features_array)[0]

    # Normalize raw_score to a 0-1 risk.
    # Example: map -0.5 to 1.0 (high risk) and 0.5 to 0.0 (low risk)
    # This mapping needs careful tuning in a real system.
    # For prototype: linear scale from -0.5 to 0.5
    min_raw_score = -0.5
    max_raw_score = 0.5
    risk_score = 1 - ((raw_score - min_raw_score) / (max_raw_score - min_raw_score))
    risk_score = np.clip(risk_score, 0.0, 1.0) # Clamp between 0 and 1

    return float(risk_score)

def get_risk_score(current_features, user_profile=None):
    """
    Determines the overall risk score for a session.
    Can combine rule-based and ML-based scores.
    """
    if not current_features:
        return 0.5 # Default to moderate if no features

    # Option 1: Pure rule-based (simple for initial prototype)
    if user_profile:
        risk_score = calculate_rule_based_risk_score(current_features, user_profile)
    else:
        # If no profile, treat as higher risk for a new/unknown user
        risk_score = 0.6 # Moderate-to-high for unprofiled sessions

    # Option 2: Pure ML-based (more advanced)
    # risk_score = calculate_ml_based_risk_score(current_features)

    # Option 3: Combine (e.g., weighted average of both)
    # For a hybrid approach, combine scores
    # if user_profile:
    #     rule_score = calculate_rule_based_risk_score(current_features, user_profile)
    #     ml_score = calculate_ml_based_risk_score(current_features)
    #     risk_score = (rule_score * 0.4) + (ml_score * 0.6) # Example weighting
    # else:
    #     risk_score = calculate_ml_based_risk_score(current_features) # Only ML if no profile

    return risk_score

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    _load_model() # Initialize the dummy model

    print("\n--- Testing Anomaly Detection ---")

    # Example: Features for a 'normal' session
    normal_features = {
        'avg_dwell_time_ms': 100.0, 'std_dwell_time_ms': 10.0, 'avg_flight_time_ms': 200.0, 'std_flight_time_ms': 20.0, 'typing_speed_cps': 5.0,
        'mouse_total_movements': 100, 'mouse_total_clicks': 5, 'mouse_total_path_length': 10000.0,
        'mouse_avg_speed_px_per_s': 200.0, 'mouse_avg_angle_change_rad': 0.5, 'mouse_std_angle_change_rad': 0.1,
        'session_duration_ms': 30000.0
    }

    # Example: Features for an 'anomalous' session (e.g., very fast typing, erratic mouse)
    anomalous_features = {
        'avg_dwell_time_ms': 50.0, # Faster
        'std_dwell_time_ms': 50.0, # More erratic
        'avg_flight_time_ms': 80.0, # Faster
        'std_flight_time_ms': 80.0,
        'typing_speed_cps': 15.0, # Very fast
        'mouse_total_movements': 500, # More movements
        'mouse_total_clicks': 15, # More clicks
        'mouse_total_path_length': 40000.0,
        'mouse_avg_speed_px_per_s': 800.0,
        'mouse_avg_angle_change_rad': 1.5, # More erratic
        'mouse_std_angle_change_rad': 0.5,
        'session_duration_ms': 20000.0
    }

    # Simulate a user profile (e.g., from user_profiler)
    user_profile_for_test = {
        'avg_dwell_time_ms': 105.0, 'std_dwell_time_ms': 11.0, 'avg_flight_time_ms': 205.0, 'std_flight_time_ms': 21.0, 'typing_speed_cps': 5.2,
        'mouse_total_movements': 105, 'mouse_total_clicks': 5, 'mouse_total_path_length': 10500.0,
        'mouse_avg_speed_px_per_s': 205.0, 'mouse_avg_angle_change_rad': 0.5, 'mouse_std_angle_change_rad': 0.1,
        'session_duration_ms': 31000.0
    }

    # Test with normal session
    risk1 = get_risk_score(normal_features, user_profile_for_test)
    print(f"Risk score for normal session: {risk1:.2f}") # Should be low

    # Test with anomalous session
    risk2 = get_risk_score(anomalous_features, user_profile_for_test)
    print(f"Risk score for anomalous session: {risk2:.2f}") # Should be high

    # Test with no user profile (new user)
    risk3 = get_risk_score(normal_features, None)
    print(f"Risk score for unprofiled user (normal behavior): {risk3:.2f}") # Should be moderate baseline