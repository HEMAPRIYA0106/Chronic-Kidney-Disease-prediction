from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
import joblib
import numpy as np
from history import PredictionHistory
from users import check_credentials

app        = Flask(__name__)
app.secret_key = "nephroai-secret-key-change-this"   # ← change to any random string
CORS(app)

model   = joblib.load("kidney_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler  = joblib.load("scaler.pkl")

ph = PredictionHistory()


# ── Login required decorator ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.is_json or request.method == "POST":
                return jsonify({"error": "Unauthorised"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ── Login page ────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET"])
def login_page():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    return render_template("login.html")


# ── Login submit ──────────────────────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login_submit():
    data     = request.get_json()
    username = data.get("username", "")
    password = data.get("password", "")
    if check_credentials(username, password):
        session["logged_in"] = True
        session["username"]  = username
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid username or password"}), 401


# ── Logout ────────────────────────────────────────────────────────────────────
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


# ── Serve main UI (protected) ─────────────────────────────────────────────────
@app.route("/")
@login_required
def home():
    return render_template("index.html")


# ── Predict + store (protected) ───────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@login_required
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


# ── History routes (all protected) ───────────────────────────────────────────
@app.route("/history", methods=["GET"])
@login_required
def get_history():
    return jsonify(ph.get_all())

@app.route("/history/ckd", methods=["GET"])
@login_required
def get_ckd():
    return jsonify(ph.get_ckd_records())

@app.route("/history/nonckd", methods=["GET"])
@login_required
def get_nonckd():
    return jsonify(ph.get_non_ckd_records())

@app.route("/history/<int:record_id>", methods=["GET"])
@login_required
def get_by_id(record_id):
    record = ph.get_by_id(record_id)
    if record:
        return jsonify(record)
    return jsonify({"error": "Record not found"}), 404

@app.route("/history/<int:record_id>", methods=["DELETE"])
@login_required
def delete_record(record_id):
    if ph.delete_by_id(record_id):
        return jsonify({"message": "Deleted", "id": record_id})
    return jsonify({"error": "Record not found"}), 404

@app.route("/history/clear", methods=["DELETE"])
@login_required
def clear_all():
    ph.clear_all()
    return jsonify({"message": "All records cleared"})

@app.route("/history/stats", methods=["GET"])
@login_required
def stats():
    return jsonify(ph.get_stats())


if __name__ == "__main__":
    app.run(debug=True)