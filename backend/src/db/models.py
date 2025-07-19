# backend/src/db/models.py

from datetime import datetime

# This is a simple data structure (not tied to any ODM)
class SessionLog:
    def __init__(self, user_id, risk_score, decision, processed_features, raw_events=None, timestamp=None):
        self.user_id = user_id
        self.timestamp = timestamp or datetime.utcnow()
        self.risk_score = risk_score
        self.decision = decision
        self.processed_features = processed_features
        self.raw_events = raw_events or []

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "risk_score": self.risk_score,
            "decision": self.decision,
            "processed_features": self.processed_features,
            "raw_events": self.raw_events
        }
