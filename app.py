from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
from history import PredictionHistory

app = Flask(__name__)
CORS(app)

model   = joblib.load("kidney_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler  = joblib.load("scaler.pkl")

ph = PredictionHistory()


# ── Serve main UI ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


# ── Predict + store ───────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data         = request.get_json()
        features     = np.array(data["features"], dtype=float).reshape(1, -1)
        processed    = imputer.transform(features)
        processed    = scaler.transform(processed)
        prediction   = int(model.predict(processed)[0])
        patient_name = data.get("patient_name", "Unknown")
        patient_id   = data.get("patient_id",   "N/A")
        record = ph.add_record(data["features"], prediction,
                               patient_name=patient_name,
                               patient_id=patient_id)
        return jsonify({"prediction": prediction, "id": record["id"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── History routes ────────────────────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(ph.get_all())

@app.route("/history/ckd", methods=["GET"])
def get_ckd():
    return jsonify(ph.get_ckd_records())

@app.route("/history/nonckd", methods=["GET"])
def get_nonckd():
    return jsonify(ph.get_non_ckd_records())

@app.route("/history/<int:record_id>", methods=["GET"])
def get_by_id(record_id):
    record = ph.get_by_id(record_id)
    if record:
        return jsonify(record)
    return jsonify({"error": "Record not found"}), 404

@app.route("/history/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    if ph.delete_by_id(record_id):
        return jsonify({"message": "Deleted", "id": record_id})
    return jsonify({"error": "Record not found"}), 404

@app.route("/history/clear", methods=["DELETE"])
def clear_all():
    ph.clear_all()
    return jsonify({"message": "All records cleared"})

@app.route("/history/stats", methods=["GET"])
def stats():
    return jsonify(ph.get_stats())


if __name__ == "__main__":
    app.run(debug=True)