# backend/src/profiling/user_profiler.py
# Changed import from postgres_connector to mongo_connector
from ..db.mongo_connector import get_mongo_client
import time # For last_updated timestamp

# Define the set of features we expect in a user profile
PROFILE_FEATURES = [
    'avg_dwell_time_ms', 'std_dwell_time_ms', 'avg_flight_time_ms', 'std_flight_time_ms', 'typing_speed_cps',
    'mouse_total_movements', 'mouse_total_clicks', 'mouse_total_path_length',
    'mouse_avg_speed_px_per_s', 'mouse_avg_angle_change_rad', 'mouse_std_angle_change_rad',
    'session_duration_ms'
]

def get_user_profile(user_id):
    """Fetches a user's behavioral profile from MongoDB."""
    db = get_mongo_client() # Changed from get_postgres_connection() to get_mongo_client()
    if db is None:
        return None

    try:
        # Find one document where 'user_id' matches
        profile = db.user_profiles_collection.find_one({"user_id": user_id})
        # Remove MongoDB's internal _id if you don't need it in the profile object
        if profile and '_id' in profile:
            del profile['_id']
        return profile
    except Exception as e:
        print(f"Error fetching user profile for {user_id} from MongoDB: {e}")
        return None

def update_user_profile(user_id, new_features):
    """
    Updates or creates a user's behavioral profile in MongoDB.
    Uses a simple weighted average for prototype.
    """
    db = get_mongo_client() # Changed from get_postgres_connection() to get_mongo_client()
    if db is None:
        return False

    try:
        # Fetch current profile to calculate updated averages
        current_profile = get_user_profile(user_id) # Uses the function defined above

        update_data = {"user_id": user_id, "last_updated": int(time.time() * 1000)}

        if current_profile:
            # Update existing profile (simple weighted average)
            for feature_name in PROFILE_FEATURES:
                current_val = current_profile.get(feature_name, 0.0)
                new_val = new_features.get(feature_name, 0.0)
                # Example: 90% old data, 10% new data for continuous learning
                update_data[feature_name] = (current_val * 0.9) + (new_val * 0.1)
        else:
            # Create new profile: just use the new_features
            for feature_name in PROFILE_FEATURES:
                update_data[feature_name] = new_features.get(feature_name, 0.0)

        # Use update_one with upsert=True to create if not exists, or update if it does
        result = db.user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": update_data},
            upsert=True
        )
        if result.upserted_id:
            print(f"New user profile for {user_id} created in MongoDB.")
        elif result.modified_count > 0:
            print(f"User profile for {user_id} updated in MongoDB.")
        else:
            print(f"User profile for {user_id} already up-to-date or no changes.")
        return True
    except Exception as e:
        print(f"Error updating/creating user profile for {user_id} in MongoDB: {e}")
        return False

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    # No PostgreSQL table creation needed anymore
    print("\n--- Testing User Profiling (MongoDB) ---")
    test_user_id = "test_mongo_user_123"

    initial_features = {
        'avg_dwell_time_ms': 100.0, 'std_dwell_time_ms': 10.0, 'avg_flight_time_ms': 200.0, 'std_flight_time_ms': 20.0, 'typing_speed_cps': 5.0,
        'mouse_total_movements': 100, 'mouse_total_clicks': 5, 'mouse_total_path_length': 10000.0,
        'mouse_avg_speed_px_per_s': 200.0, 'mouse_avg_angle_change_rad': 0.5, 'mouse_std_angle_change_rad': 0.1,
        'session_duration_ms': 30000.0
    }

    print(f"Updating/Creating profile for {test_user_id} with initial features...")
    update_user_profile(test_user_id, initial_features)

    profile = get_user_profile(test_user_id)
    if profile:
        print(f"\nProfile for {test_user_id}:\n{profile}")
    else:
        print(f"No profile found for {test_user_id}.")

    # Simulate new session features (e.g., slightly different)
    new_session_features = {
        'avg_dwell_time_ms': 110.0, 'std_dwell_time_ms': 12.0, 'avg_flight_time_ms': 210.0, 'std_flight_time_ms': 22.0, 'typing_speed_cps': 5.5,
        'mouse_total_movements': 110, 'mouse_total_clicks': 6, 'mouse_total_path_length': 11000.0,
        'mouse_avg_speed_px_per_s': 210.0, 'mouse_avg_angle_change_rad': 0.6, 'mouse_std_angle_change_rad': 0.12,
        'session_duration_ms': 32000.0
    }
    print(f"\nUpdating profile for {test_user_id} with new session features...")
    update_user_profile(test_user_id, new_session_features)
    profile_updated = get_user_profile(test_user_id)
    if profile_updated:
        print(f"\nUpdated Profile for {test_user_id}:\n{profile_updated}")
    else:
        print(f"No profile found for {test_user_id}.")