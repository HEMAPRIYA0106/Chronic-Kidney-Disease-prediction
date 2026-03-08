from datetime import datetime


class PredictionHistory:

    def __init__(self):
        self.history = []
        self.next_id = 1

    # Add
    def add_record(self, features, prediction):
        record = {
            "id":         self.next_id,
            "timestamp":  datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "prediction": prediction,
            "result":     "CKD Detected" if prediction == 1 else "No CKD",
            "age":   features[0],  "bp":    features[1],
            "sg":    features[2],  "al":    features[3],
            "su":    features[4],  "bgr":   features[5],
            "bu":    features[6],  "sc":    features[7],
            "sod":   features[8],  "pot":   features[9],
            "hemo":  features[10], "pcv":   features[11],
            "rc":    features[12], "htn":   features[13],
            "dm":    features[14], "appet": features[15],
            "pe":    features[16], "ane":   features[17],
        }
        self.history.append(record)
        self.next_id += 1
        return record

    # Get all (newest first)
    def get_all(self):
        return list(reversed(self.history))

    # Get only CKD records
    def get_ckd_records(self):
        return [r for r in reversed(self.history) if r["prediction"] == 1]

    # Get only Non-CKD records
    def get_non_ckd_records(self):
        return [r for r in reversed(self.history) if r["prediction"] == 0]

    # Get by ID
    def get_by_id(self, id):
        for r in self.history:
            if r["id"] == id:
                return r
        return None

    # Delete by ID
    def delete_by_id(self, id):
        before = len(self.history)
        self.history = [r for r in self.history if r["id"] != id]
        return len(self.history) < before

    # Clear all 
    def clear_all(self):
        self.history.clear()
        self.next_id = 1

    # Total number of predictions stored 
    def size(self):
        return len(self.history)

    # CKD count 
    def ckd_count(self):
        return sum(1 for r in self.history if r["prediction"] == 1)

    # Non-CKD count 
    def non_ckd_count(self):
        return sum(1 for r in self.history if r["prediction"] == 0)

    # Stats (all counts in one call) 
    def get_stats(self):
        return {
            "total":   self.size(),
            "ckd":     self.ckd_count(),
            "non_ckd": self.non_ckd_count()
        }
