import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from sklearn.inspection import permutation_importance

from db import (
    init_db,
    create_user, get_user_by_username, get_user_by_id, list_users,
    add_check, get_recent_checks, clear_checks,
    analytics_summary, save_model_meta, latest_model_meta
)

# ---------------- App setup ----------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "upi_data.csv")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model
model = joblib.load(MODEL_PATH)

# ---------------- Explainable AI helpers ----------------
FEATURES = ["amount", "hour", "new_device", "transaction_count"]
FEATURE_LABELS = {
    "amount": "Amount (₹)",
    "hour": "Hour",
    "new_device": "New Device",
    "transaction_count": "Transactions Today"
}

def load_reference_data():
    """Load upi_data.csv (if exists) to compute feature importances."""
    if not os.path.exists(DATASET_PATH):
        return None, None

    df = pd.read_csv(DATASET_PATH)

    label_col = None
    for c in ["fraud", "label", "is_fraud", "target", "Class"]:
        if c in df.columns:
            label_col = c
            break

    if not label_col:
        return None, None

    if not all(c in df.columns for c in FEATURES):
        return None, None

    X = df[FEATURES]
    y = df[label_col].astype(int)
    return X, y

def get_feature_importance(model_obj, X_ref, y_ref):
    """
    Returns list of (feature, importance_score) sorted desc.
    Uses feature_importances_ if available; else permutation_importance.
    """
    if hasattr(model_obj, "feature_importances_"):
        imp = list(zip(FEATURES, model_obj.feature_importances_))
        imp.sort(key=lambda x: x[1], reverse=True)
        return imp

    r = permutation_importance(model_obj, X_ref, y_ref, n_repeats=5, random_state=42)
    imp = list(zip(FEATURES, r.importances_mean))
    imp.sort(key=lambda x: x[1], reverse=True)
    return imp

# ---------------- Core helpers ----------------
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return get_user_by_id(uid)

def login_required():
    if not current_user():
        return redirect(url_for("login"))
    return None

def admin_required():
    u = current_user()
    if not u or int(u["is_admin"]) != 1:
        return redirect(url_for("home"))
    return None

def predict_df(df):
    X = df[FEATURES]
    preds = model.predict(X).astype(int)
    probs = (model.predict_proba(X)[:, 1] * 100).round(2)
    return preds, probs

def predict_one(amount, hour, new_device, transaction_count):
    df = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "new_device": new_device,
        "transaction_count": transaction_count
    }])
    preds, probs = predict_df(df)
    return int(preds[0]), float(probs[0])

# ---------------- Auth routes ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        if not username or not password:
            return render_template("register.html", error="Username and password required.")

        if get_user_by_username(username):
            return render_template("register.html", error="Username already exists.")

        # IMPORTANT: pbkdf2 to avoid Mac hashlib.scrypt issue
        pw_hash = generate_password_hash(password, method="pbkdf2:sha256")

        create_user(
            username,
            pw_hash,
            datetime.now().strftime("%d-%b-%Y %I:%M %p"),
            is_admin=0
        )
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")  # ✅ Rate limit login only
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = get_user_by_username(username)
        if not user or not check_password_hash(user["password_hash"], password):
            return render_template("login.html", error="Invalid credentials.")

        session["user_id"] = int(user["id"])
        return redirect(url_for("home"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- Pages ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    guard = login_required()
    if guard:
        return guard

    u = current_user()
    result = None
    prob = None
    label = None

    if request.method == "POST":
        amount = float(request.form["amount"])
        hour = int(request.form["hour"])
        device = int(request.form["new_device"])
        count = int(request.form["transaction_count"])

        pred, probability = predict_one(amount, hour, device, count)
        prob = round(probability, 2)

        if pred == 1:
            result = "Fraud Transaction Detected"
            label = "fraud"
        else:
            result = "Safe Transaction"
            label = "safe"

        created_at = datetime.now().strftime("%d-%b-%Y %I:%M %p")
        add_check(int(u["id"]), created_at, amount, hour, device, count, prob, pred)

    rows = get_recent_checks(user_id=int(u["id"]), limit=20)
    history = [{
        "time": r["created_at"],
        "amount": r["amount"],
        "hour": r["hour"],
        "device": "New" if r["new_device"] == 1 else "Old",
        "count": r["transaction_count"],
        "prob": r["fraud_prob"],
        "status": "Fraud Transaction Detected" if r["label"] == 1 else "Safe Transaction"
    } for r in rows]

    return render_template("index.html", user=u, result=result, prob=prob, label=label, history=history)

@app.route("/explain", methods=["GET", "POST"])
def explain():
    guard = login_required()
    if guard:
        return guard

    u = current_user()
    explanation = None
    error = None

    if request.method == "POST":
        try:
            amount = float(request.form["amount"])
            hour = int(request.form["hour"])
            new_device = int(request.form["new_device"])
            transaction_count = int(request.form["transaction_count"])

            pred, prob = predict_one(amount, hour, new_device, transaction_count)

            X_ref, y_ref = load_reference_data()

            # Reasons
            reasons = []
            if X_ref is not None:
                imp = get_feature_importance(model, X_ref, y_ref)  # list of (feature, score)
                # take top 3 features
                top = imp[:3]
                for f, score in top:
                    reasons.append(f"{FEATURE_LABELS.get(f, f)} is important (score: {round(float(score), 4)})")
            else:
                # fallback heuristic reasons
                if amount >= 5000: reasons.append("High transaction amount")
                if hour <= 5 or hour >= 23: reasons.append("Unusual transaction time")
                if new_device == 1: reasons.append("New device used")
                if transaction_count >= 8: reasons.append("High number of transactions today")
                if not reasons:
                    reasons = ["Normal pattern based on available features"]

            explanation = {
                "pred_label": "Fraud" if pred == 1 else "Safe",
                "prob": round(prob, 2),
                "input": {
                    "amount": amount,
                    "hour": hour,
                    "new_device": new_device,
                    "transaction_count": transaction_count
                },
                "reasons": reasons
            }

        except Exception as e:
            error = f"Explain error: {e}"

    return render_template("explain.html", user=u, explanation=explanation, error=error)

@app.route("/about")
def about():
    u = current_user()
    meta = latest_model_meta()

    metrics = None
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            label_col = None
            for c in ["fraud", "label", "is_fraud", "target", "Class"]:
                if c in df.columns:
                    label_col = c
                    break

            if label_col and all(c in df.columns for c in FEATURES):
                X = df[FEATURES]
                y = df[label_col].astype(int)
                y_pred = model.predict(X).astype(int)

                acc = float((y_pred == y).mean() * 100)
                tp = int(((y_pred == 1) & (y == 1)).sum())
                tn = int(((y_pred == 0) & (y == 0)).sum())
                fp = int(((y_pred == 1) & (y == 0)).sum())
                fn = int(((y_pred == 0) & (y == 1)).sum())

                metrics = {"accuracy": round(acc, 2), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "label_col": label_col}
        except Exception:
            metrics = None

    return render_template("about.html", user=u, metrics=metrics, meta=meta)

@app.route("/dashboard")
def dashboard():
    guard = login_required()
    if guard:
        return guard
    u = current_user()
    summary = analytics_summary(user_id=int(u["id"]))
    return render_template("dashboard.html", user=u, summary=summary)

# ---------------- Clear History ----------------
@app.route("/clear-history", methods=["POST"])
def clear_history_route():
    guard = login_required()
    if guard:
        return ("", 401)
    u = current_user()
    clear_checks(user_id=int(u["id"]))
    return ("", 204)

# ---------------- Batch CSV ----------------
@app.route("/batch", methods=["GET", "POST"])
def batch():
    guard = login_required()
    if guard:
        return guard

    u = current_user()
    results = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file uploaded."
        else:
            f = request.files["file"]
            if f.filename == "":
                error = "Please choose a CSV file."
            else:
                try:
                    df = pd.read_csv(f)
                    missing = [c for c in FEATURES if c not in df.columns]
                    if missing:
                        error = f"Missing columns: {', '.join(missing)}"
                    else:
                        preds, probs = predict_df(df)
                        out = df.copy()
                        out["fraud_prob_%"] = probs
                        out["prediction"] = np.where(preds == 1, "Fraud", "Safe")

                        now = datetime.now().strftime("%d-%b-%Y %I:%M %p")
                        for i in range(min(len(out), 200)):
                            add_check(
                                int(u["id"]), now,
                                float(out.loc[i, "amount"]),
                                int(out.loc[i, "hour"]),
                                int(out.loc[i, "new_device"]),
                                int(out.loc[i, "transaction_count"]),
                                float(out.loc[i, "fraud_prob_%"]),
                                1 if out.loc[i, "prediction"] == "Fraud" else 0
                            )

                        path = os.path.join(BASE_DIR, "last_batch_results.csv")
                        out.to_csv(path, index=False)
                        results = out.head(100).to_dict(orient="records")
                except Exception as e:
                    error = f"Error: {e}"

    return render_template("batch.html", user=u, results=results, error=error)

@app.route("/download-batch")
def download_batch():
    guard = login_required()
    if guard:
        return redirect(url_for("login"))
    path = os.path.join(BASE_DIR, "last_batch_results.csv")
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("batch"))

# ---------------- REST API ----------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    if not all(k in data for k in FEATURES):
        return jsonify({"error": f"Missing required fields: {FEATURES}"}), 400

    pred, prob = predict_one(
        float(data["amount"]),
        int(data["hour"]),
        int(data["new_device"]),
        int(data["transaction_count"])
    )
    return jsonify({
        "prediction": int(pred),
        "label": "Fraud" if pred == 1 else "Safe",
        "fraud_probability_percent": round(prob, 2)
    })

@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    u = current_user()
    if u:
        s = analytics_summary(user_id=int(u["id"]))
    else:
        s = analytics_summary(user_id=None)
    return jsonify(s)

# ---------------- Admin ----------------
@app.route("/admin")
def admin():
    guard = admin_required()
    if guard:
        return guard
    users = list_users()
    checks = get_recent_checks(user_id=None, limit=50)
    return render_template("admin.html", user=current_user(), users=users, checks=checks)

# ---------------- Retrain (Admin only) ----------------
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    guard = admin_required()
    if guard:
        return guard

    message = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file uploaded."
        else:
            f = request.files["file"]
            if f.filename == "":
                error = "Please choose a CSV file."
            else:
                try:
                    filename = secure_filename(f.filename)
                    saved_path = os.path.join(UPLOAD_DIR, filename)
                    f.save(saved_path)

                    df = pd.read_csv(saved_path)

                    label_col = None
                    for c in ["fraud", "label", "is_fraud", "target", "Class"]:
                        if c in df.columns:
                            label_col = c
                            break

                    if not label_col:
                        error = "No label column found. Add one of: fraud/label/is_fraud/target/Class"
                    elif not all(c in df.columns for c in FEATURES):
                        error = f"CSV must contain: {', '.join(FEATURES)}"
                    else:
                        from sklearn.model_selection import train_test_split
                        from sklearn.ensemble import RandomForestClassifier

                        X = df[FEATURES]
                        y = df[label_col].astype(int)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42,
                            stratify=y if y.nunique() > 1 else None
                        )

                        new_model = RandomForestClassifier(n_estimators=200, random_state=42)
                        new_model.fit(X_train, y_train)
                        acc = float((new_model.predict(X_test) == y_test).mean() * 100)

                        # Backup old model
                        backup = os.path.join(BASE_DIR, f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                        if os.path.exists(MODEL_PATH):
                            os.replace(MODEL_PATH, backup)

                        joblib.dump(new_model, MODEL_PATH)

                        # Reload model globally
                        global model
                        model = joblib.load(MODEL_PATH)

                        save_model_meta(
                            created_at=datetime.now().strftime("%d-%b-%Y %I:%M %p"),
                            accuracy=round(acc, 2),
                            label_col=label_col,
                            notes=f"Retrained from {filename}"
                        )

                        message = f"✅ Retrained model successfully. Test accuracy: {round(acc,2)}%. Backup: {os.path.basename(backup)}"
                except Exception as e:
                    error = f"Retrain failed: {e}"

    return render_template("retrain.html", user=current_user(), message=message, error=error)

# ---------------- Main ----------------
if __name__ == "__main__":
    init_db()

    # Create default admin (admin / admin123) if not exists
    if not get_user_by_username("admin"):
        create_user(
            "admin",
            generate_password_hash("admin123", method="pbkdf2:sha256"),
            datetime.now().strftime("%d-%b-%Y %I:%M %p"),
            is_admin=1
        )

    print("Model loaded from:", MODEL_PATH)
    app.run(debug=True, port=int(os.getenv("PORT", "5001")))