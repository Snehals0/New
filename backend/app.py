from flask import Flask, jsonify, request, redirect, url_for, flash
from flask_cors import CORS
import uuid
import time
import json
from datetime import datetime

# --- Flask-Admin Imports ---
from flask_admin import Admin, AdminIndexView, BaseView, expose
from flask_admin.contrib.pymongo import ModelView
# Corrected import for filter types specifically for PyMongo
from flask_admin.contrib.pymongo import filters as pymongo_filters # <-- CORRECTED IMPORT


# --- WTForms Import ---
from wtforms import Form # Import generic Form class from WTForms
from wtforms.fields import StringField, IntegerField, FloatField # Import specific field types for column_type_map

# --- MongoDB Direct Client Imports ---
from pymongo import MongoClient
from bson.objectid import ObjectId



# --- Relative Imports for Project Modules ---
from .src.db.mongo_connector import get_mongo_client
from .src.data_processing.feature_extractor import process_raw_data
from .src.profiling.user_profiler import get_user_profile, update_user_profile
from .src.ml_models.anomaly_detector import get_risk_score, _load_model

# --- Flask Application Initialization ---
app = Flask(__name__)
CORS(app)
app.config.from_object('backend.config.db_config')
app.config['SECRET_KEY'] = 'your_super_secret_key_for_admin_session_replace_this_in_prod'
app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'

# Initialize Direct PyMongo Client for Flask-Admin
mongo_client_admin = MongoClient(app.config['MONGO_URI'])
db_admin = mongo_client_admin[app.config['MONGO_DB_NAME']]

# --- Flask-Admin Setup ---

# Basic Admin Authentication (for prototype)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password'

class MyAdminIndexView(AdminIndexView):
    def is_accessible(self):
        return 'logged_in' in request.cookies and request.cookies.get('logged_in') == 'true'

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('admin_login'))

admin = Admin(app, name='Banking Security Admin', template_mode='bootstrap3', index_view=MyAdminIndexView())

# --- Define Flask-Admin ModelViews for PyMongo ---
# Override get_form() to return a basic Form class when auto-scaffolding is not desired.
# Explicitly define column_filters using filters from flask_admin.contrib.pymongo.filters.

class RawBehavioralLogView(ModelView):
    """Admin view for the 'raw_behavioral_logs' collection."""
    column_list = ('_id', 'session_id', 'user_id', 'frontend_timestamp', 'server_received_at', 'encrypted_data')
    column_searchable_list = ('session_id', 'user_id')
    # Explicitly define filters using pymongo_filters
    column_filters = [
        pymongo_filters.FilterLike('user_id', name='User ID (contains)'), # <-- USED CORRECTED IMPORT
        pymongo_filters.FilterEqual('user_id', name='User ID (exact)'),   # <-- USED CORRECTED IMPORT
        pymongo_filters.FilterGreater('server_received_at', name='Received After (timestamp)'), # <-- USED CORRECTED IMPORT
        pymongo_filters.FilterSmaller('server_received_at', name='Received Before (timestamp)') # <-- USED CORRECTED IMPORT
    ]
    can_create = False
    can_edit = False
    can_delete = True
    form_columns = ('session_id', 'user_id', 'frontend_timestamp', 'server_received_at', 'encrypted_data')
    form_create_columns = []
    form_edit_columns = []

    def get_form(self):
        return Form

class UserProfileView(ModelView):
    """Admin view for the 'user_profiles_collection'."""
    column_list = ('_id', 'user_id', 'avg_dwell_time_ms', 'typing_speed_cps', 'mouse_avg_speed_px_per_s', 'last_updated')
    column_searchable_list = ('user_id',)
    # Explicitly define filters
    column_filters = [
        pymongo_filters.FilterLike('user_id', name='User ID (contains)'),
        pymongo_filters.FilterEqual('user_id', name='User ID (exact)'),
        pymongo_filters.FilterGreater('last_updated', name='Updated After (timestamp)'),
        pymongo_filters.FilterSmaller('last_updated', name='Updated Before (timestamp)'),
        pymongo_filters.FilterGreater('typing_speed_cps', name='Typing Speed (>)'),
        pymongo_filters.FilterSmaller('typing_speed_cps', name='Typing Speed (<)')
    ]
    can_create = False
    can_delete = True
    form_columns = ('user_id', 'avg_dwell_time_ms', 'std_dwell_time_ms', 'avg_flight_time_ms',
                    'std_flight_time_ms', 'typing_speed_cps', 'mouse_total_movements',
                    'mouse_total_clicks', 'mouse_total_path_length', 'mouse_avg_speed_px_per_s',
                    'mouse_avg_angle_change_rad', 'mouse_std_angle_change_rad', 'session_duration_ms', 'last_updated')
    form_create_columns = []
    form_edit_columns = ('avg_dwell_time_ms', 'std_dwell_time_ms', 'avg_flight_time_ms',
                        'std_flight_time_ms', 'typing_speed_cps', 'mouse_total_movements',
                        'mouse_total_clicks', 'mouse_total_path_length', 'mouse_avg_speed_px_per_s',
                        'mouse_avg_angle_change_rad', 'mouse_std_angle_change_rad', 'session_duration_ms', 'last_updated')
    def get_form(self):
        return Form

class SessionLogView(ModelView):
    column_list = ('_id', 'session_id', 'user_id', 'timestamp', 'risk_score', 'action_taken', 'raw_data_mongo_id', 'processed_features')
    column_searchable_list = ('session_id', 'user_id', 'action_taken')
    # Explicitly define filters
    column_filters = [
        pymongo_filters.FilterLike('user_id', name='User ID (contains)'),
        pymongo_filters.FilterEqual('user_id', name='User ID (exact)'),
        pymongo_filters.FilterEqual('action_taken', name='Action Taken', options=[('allow', 'Allow'), ('require_2fa', 'Require 2FA'), ('deny_access', 'Deny Access')]),
        pymongo_filters.FilterGreater('risk_score', name='Risk Score (>)'),
        pymongo_filters.FilterSmaller('risk_score', name='Risk Score (<)'),
        pymongo_filters.FilterGreater('timestamp', name='Timestamp After (ms)'),
        pymongo_filters.FilterSmaller('timestamp', name='Timestamp Before (ms)')
    ]
    can_create = False
    can_edit = False
    can_delete = True
    form_columns = ('session_id', 'user_id', 'timestamp', 'risk_score', 'action_taken', 'raw_data_mongo_id', 'processed_features')
    form_create_columns = []
    form_edit_columns = []

    def get_form(self):
        return Form

# Add the defined ModelViews to the Flask-Admin instance, linking them to their respective PyMongo collections
admin.add_view(RawBehavioralLogView(db_admin.raw_behavioral_logs, name='Raw Logs'))
admin.add_view(UserProfileView(db_admin.user_profiles_collection, name='User Profiles'))
admin.add_view(SessionLogView(db_admin.session_logs_collection, name='Session Logs'))
# --- Alert View for High-Risk Triggers ---
class AlertLogView(ModelView):
    column_list = ('_id', 'user_id', 'session_id', 'risk_score', 'reason', 'timestamp', 'action_taken')
    column_searchable_list = ('user_id', 'reason', 'action_taken')
    can_create = False
    can_edit = False
    can_delete = True
    def get_form(self):
        return Form

admin.add_view(AlertLogView(db_admin.alert_logs, name='Alerts'))



# --- Admin Login/Logout Routes ---
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            response = redirect(url_for('admin.index'))
            response.set_cookie('logged_in', 'true', max_age=3600)
            return response
        else:
            flash('Invalid credentials. Please try again.', 'error')
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Admin Login</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <style>
                body { display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #f8f9fa; }
                .login-container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 350px; }
                .form-control { margin-bottom: 15px; }
                .btn-primary { width: 100%; }
                .alert { margin-top: 15px; }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h3 class="text-center">Admin Login</h3>
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                <form method="POST">
                    <div class="form-group">
                        <input type="text" name="username" class="form-control" placeholder="Username" required>
                    </div>
                    <div class="form-group">
                        <input type="password" name="password" class="form-control" placeholder="Password" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Login</button>
                </form>
            </div>
        </body>
        </html>
    '''

@app.route('/admin-logout')
def admin_logout():
    response = redirect(url_for('admin_login'))
    response.set_cookie('logged_in', '', expires=0)
    flash('You have been logged out.', 'info')
    return response

# --- Flask Routes (Existing) ---

@app.route('/')
def hello_world():
    return jsonify(message="Backend says hello from Behavioral Biometrics AI (MongoDB Only)!")

@app.route('/api/collect_behavior', methods=['POST'])
def collect_behavior():
    session_id = str(uuid.uuid4())
    current_timestamp_ms = int(time.time() * 1000)

    if not request.is_json:
        print(f"[{session_id}] Received request is not JSON. Headers: {request.headers}")
        return jsonify(status="error", message="Request must be JSON"), 400

    data = request.json
    user_id = data.get('userId')
    session_data_encrypted_b64 = data.get('sessionData')
    frontend_timestamp = data.get('timestamp')

    if not user_id or not session_data_encrypted_b64:
        print(f"[{session_id}] Missing userId or sessionData in request.")
        return jsonify(status="error", message="Missing userId or sessionData"), 400

    print(f"\n--- Processing Session: {session_id} for User: {user_id} ---")

    mongo_db_direct = get_mongo_client()
    if mongo_db_direct is None:
        print(f"[{session_id}] MongoDB connection failed. Cannot proceed with data storage.")
        return jsonify(status="error", message="Database connection error."), 500

    raw_data_mongo_id = None
    try:
        insert_result = mongo_db_direct.raw_behavioral_logs.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "frontend_timestamp": frontend_timestamp,
            "server_received_at": current_timestamp_ms,
            "encrypted_data": session_data_encrypted_b64
        })
        raw_data_mongo_id = str(insert_result.inserted_id)
        print(f"[{session_id}] Raw data stored in MongoDB with ID: {raw_data_mongo_id}")
    except Exception as e:
        print(f"[{session_id}] Error storing raw data in MongoDB: {e}")

    processed_features = process_raw_data(session_data_encrypted_b64)
    if not processed_features:
        print(f"[{session_id}] No features extracted from raw data, returning default risk.")
        return jsonify(status="error", message="Failed to extract features.", risk_score=0.5, action="allow"), 500

    print(f"[{session_id}] Extracted features:")
    print(json.dumps(processed_features, indent=2))

    user_profile = get_user_profile(user_id)
    if not user_profile:
        print(f"[{session_id}] No existing profile for {user_id}. Treating as a new user for profiling.")
    else:
        print(f"[{session_id}] User profile for {user_id} loaded.")

    risk_score = get_risk_score(processed_features, user_profile)
    print(f"[{session_id}] Calculated risk score: {risk_score:.4f}")

    action_taken = "allow"
    if risk_score >= 0.8:
        action_taken = "deny_access"
        print(f"[{session_id}] High risk detected for {user_id}. Action: Denying access.")

        # ✅ Save alert to alert_logs collection
        try:
            mongo_db_direct.alert_logs.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "risk_score": risk_score,
                "reason": "High risk behavior pattern detected",
                "timestamp": current_timestamp_ms,
                "action_taken": action_taken
            })
            print(f"[{session_id}] 🚨 Alert saved to alert_logs collection.")
        except Exception as e:
            print(f"[{session_id}] Failed to save alert: {e}")

    elif risk_score >= 0.5:
        action_taken = "require_2fa"
        print(f"[{session_id}] Moderate risk detected for {user_id}. Action: Requiring 2FA.")
    else:
        print(f"[{session_id}] Low risk detected for {user_id}. Action: Allowing session.")

    update_successful = update_user_profile(user_id, processed_features)
    if not update_successful:
        print(f"[{session_id}] Failed to update user profile for {user_id}.")

    try:
        mongo_db_direct.session_logs_collection.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": current_timestamp_ms,
            "risk_score": risk_score,
            "action_taken": action_taken,
            "raw_data_mongo_id": raw_data_mongo_id,
            "processed_features": processed_features
        })
        print(f"[{session_id}] Session log saved to MongoDB.")
        # Check for alert-worthy condition
        alert_needed = False
        reason = ""
        # Trigger 1: Very high individual risk
        if risk_score >= 0.9:
            alert_needed = True
            reason = "High risk score ≥ 0.9"
            # Trigger 2: Repeated high risk
        else:
            window_ms = 10 * 60 * 1000  # 10 minutes
            past = current_timestamp_ms - window_ms
            
            count = mongo_db.session_logs_collection.count_documents({
                "user_id": user_id,
                "risk_score": {"$gte": 0.8},
                "timestamp": {"$gte": past}
        })
        if count >= 3:
            alert_needed = True
            reason = f"{count} high-risk sessions in last 10 min"
            if alert_needed:
                alert = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "risk_score": risk_score,
                    "reason": reason,
                    "timestamp": current_timestamp_ms,
                    "action_taken": action_taken
            }
            mongo_db.alert_logs.insert_one(alert)
            print(f"[{session_id}] 🚨 Admin alert logged: {reason}")

    except Exception as e:
        print(f"[{session_id}] Error saving session log to MongoDB: {e}")
        

    return jsonify(
        status="success",
        session_id=session_id,
        risk_score=round(risk_score, 4),
        action=action_taken,
        message=f"Session processed. Action: {action_taken.replace('_', ' ')}"
    )

if __name__ == '__main__':
    from .src.ml_models.anomaly_detector import _load_model
    _load_model()

    app.run(debug=True, host='0.0.0.0', port=5000)