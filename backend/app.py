# backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS # Used for handling Cross-Origin Resource Sharing
import uuid # For generating unique session IDs
import time # For generating timestamps
import json # Used for JSON operations, especially for printing features

# --- Relative Imports for Project Modules ---
# These imports assume app.py is run as a module (e.g., 'python -m backend.app')
# from the project root directory.

# Import MongoDB database connector
from .src.db.mongo_connector import get_mongo_client

# Import feature engineering module
from .src.data_processing.feature_extractor import process_raw_data

# Import user profiling module
from .src.profiling.user_profiler import get_user_profile, update_user_profile

# Import anomaly detection module
from .src.ml_models.anomaly_detector import get_risk_score, _load_model

# --- Flask Application Initialization ---
app = Flask(__name__)
# Enable CORS for all routes. In production, you would restrict 'origins'
# to specific frontend domains (e.g., CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}}))
CORS(app)

# --- Flask Routes ---

@app.route('/')
def hello_world():
    """
    Root endpoint to confirm the backend is running.
    """
    return jsonify(message="Backend says hello from Behavioral Biometrics AI (MongoDB Only)!")

@app.route('/api/collect_behavior', methods=['POST'])
def collect_behavior():
    """
    Receives behavioral data from the frontend, processes it,
    performs anomaly detection, updates user profiles, and logs the session.
    """
    session_id = str(uuid.uuid4()) # Generate a unique session ID for the current interaction
    current_timestamp_ms = int(time.time() * 1000) # Server-side timestamp in milliseconds

    # 1. Input Validation and Data Extraction
    # Ensure the request body is JSON. If not, return a 400 Bad Request.
    if not request.is_json:
        print(f"[{session_id}] Received request is not JSON. Headers: {request.headers}")
        return jsonify(status="error", message="Request must be JSON"), 400

    data = request.json # Parse the JSON body of the request
    user_id = data.get('userId') # Get user ID from the request payload
    session_data_encrypted_b64 = data.get('sessionData') # Get encrypted behavioral data (Base64 string)
    frontend_timestamp = data.get('timestamp') # Get timestamp from the frontend

    # Validate essential fields
    if not user_id or not session_data_encrypted_b64:
        print(f"[{session_id}] Missing userId or sessionData in request.")
        return jsonify(status="error", message="Missing userId or sessionData"), 400

    print(f"\n--- Processing Session: {session_id} for User: {user_id} ---")

    # Get MongoDB client instance
    mongo_db = get_mongo_client()
    # CRUCIAL FIX: Check if mongo_db is None, not if not mongo_db
    if mongo_db is None: # <--- THIS IS THE CORRECTED LINE
        print(f"[{session_id}] MongoDB connection failed. Cannot proceed with data storage.")
        return jsonify(status="error", message="Database connection error."), 500

    # 2. Store Raw Behavioral Data in MongoDB
    raw_data_mongo_id = None
    try:
        # Insert the raw, encrypted behavioral data into a 'raw_behavioral_logs' collection
        insert_result = mongo_db.raw_behavioral_logs.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "frontend_timestamp": frontend_timestamp,
            "server_received_at": current_timestamp_ms,
            "encrypted_data": session_data_encrypted_b64
        })
        raw_data_mongo_id = str(insert_result.inserted_id) # Get the MongoDB document ID
        print(f"[{session_id}] Raw data stored in MongoDB with ID: {raw_data_mongo_id}")
    except Exception as e:
        print(f"[{session_id}] Error storing raw data in MongoDB: {e}")
        # Log the error but continue processing if raw data storage isn't a critical failure point

    # 3. Feature Engineering
    # Process the raw encrypted data to extract meaningful behavioral features
    processed_features = process_raw_data(session_data_encrypted_b64)
    if not processed_features:
        print(f"[{session_id}] No features extracted from raw data, returning default risk.")
        return jsonify(status="error", message="Failed to extract features.", risk_score=0.5, action="allow"), 500

    print(f"[{session_id}] Extracted features:")
    print(json.dumps(processed_features, indent=2)) # Pretty-print features for debugging

    # 4. User Behavior Profiling
    # Retrieve the user's existing behavioral profile from MongoDB
    user_profile = get_user_profile(user_id)
    if not user_profile:
        print(f"[{session_id}] No existing profile for {user_id}. Treating as a new user for profiling.")
    else:
        print(f"[{session_id}] User profile for {user_id} loaded.")

    # 5. Anomaly Detection
    # Calculate a risk score for the current session based on extracted features and user profile
    risk_score = get_risk_score(processed_features, user_profile)
    print(f"[{session_id}] Calculated risk score: {risk_score:.4f}") # Print with 4 decimal places

    # 6. Risk Evaluation & Dynamic Response
    # Determine the appropriate action based on predefined risk thresholds
    action_taken = "allow"
    if risk_score >= 0.8:
        action_taken = "deny_access"
        print(f"[{session_id}] High risk detected for {user_id}. Action: Denying access.")
    elif risk_score >= 0.5:
        action_taken = "require_2fa"
        print(f"[{session_id}] Moderate risk detected for {user_id}. Action: Requiring 2FA.")
    else:
        print(f"[{session_id}] Low risk detected for {user_id}. Action: Allowing session.")

    # 7. Update User Profile for Continuous Learning
    # Update the user's historical profile with the new session's features.
    # This is done AFTER risk calculation to avoid bias in the current session.
    update_successful = update_user_profile(user_id, processed_features)
    if not update_successful:
        print(f"[{session_id}] Failed to update user profile for {user_id}.")

    # 8. Log Session Details in MongoDB
    try:
        # Log the entire session's outcome, linking to raw data and storing processed features
        mongo_db.session_logs_collection.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": current_timestamp_ms,
            "risk_score": risk_score,
            "action_taken": action_taken,
            "raw_data_mongo_id": raw_data_mongo_id, # Reference to the raw data document
            "processed_features": processed_features # Store processed features for auditing/analytics
        })
        print(f"[{session_id}] Session log saved to MongoDB.")
    except Exception as e:
        print(f"[{session_id}] Error saving session log to MongoDB: {e}")

    # Return the response to the frontend
    return jsonify(
        status="success",
        session_id=session_id,
        risk_score=round(risk_score, 4), # Return risk score rounded for cleaner display
        action=action_taken,
        message=f"Session processed. Action: {action_taken.replace('_', ' ')}"
    )

# --- Application Entry Point ---
if __name__ == '__main__':
    # Initialize the dummy ML model when the app starts
    # This import needs to be relative, similar to the ones above
    from .src.ml_models.anomaly_detector import _load_model
    _load_model()

    # Run the Flask development server
    # Set debug=True for automatic reloading during development
    # host='0.0.0.0' makes it accessible from your network, '127.0.0.1' is localhost only
    app.run(debug=True, host='0.0.0.0', port=5000)