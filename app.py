from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import uuid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
MODEL_DIR = "models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SESSIONS = {}


# ------------------------------------------------------
# 1. UPLOAD ROUTE
# ------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = str(uuid.uuid4()) + "_" + file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": "Failed to read file", "details": str(e)}), 400

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"df": df, "processed": None, "train_data": None, "metrics": None}

    return jsonify({
        "session_id": session_id,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns)
    })


# ------------------------------------------------------
# 2. PREPROCESS ROUTE
# ------------------------------------------------------
@app.route("/preprocess", methods=["POST"])
def preprocess():
    data = request.json
    session_id = data.get("session_id")
    method = data.get("method")
    target = data.get("target")

    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    df = SESSIONS[session_id]["df"]

    if target not in df.columns:
        return jsonify({"error": "Target column not found"}), 400

    X = df.drop(columns=[target])
    y = df[target]

    # Separate numeric and non-numeric columns
    numeric = X.select_dtypes(include=["number"]).fillna(0)
    non_numeric = X.select_dtypes(exclude=["number"]).fillna("")

    # Ensure numeric DataFrame always exists (even when empty)
    if numeric is None or numeric.empty:
        numeric = pd.DataFrame(index=X.index)

    # Apply scaling ONLY if numeric exists
    if numeric.shape[1] > 0:
        if method == "standard":
            scaler = StandardScaler()
            numeric = pd.DataFrame(scaler.fit_transform(numeric), columns=numeric.columns)

        elif method == "minmax":
            scaler = MinMaxScaler()
            numeric = pd.DataFrame(scaler.fit_transform(numeric), columns=numeric.columns)

    # Encode non-numeric (categorical) features
    if len(non_numeric.columns) > 0:
        non_numeric = pd.get_dummies(non_numeric, drop_first=True)

    # Combine everything
    processed = pd.concat([numeric, non_numeric], axis=1)

    # Final safety check
    if processed.shape[1] == 0:
        return jsonify({"error": "Dataset has no usable features"}), 400

    SESSIONS[session_id]["processed"] = (processed, y)

    return jsonify({
        "message": "Preprocessing done",
        "features": list(processed.columns)
    })


# ------------------------------------------------------
# 3. TRAIN-TEST SPLIT ROUTE
# ------------------------------------------------------
@app.route("/split", methods=["POST"])
def split():
    data = request.json
    session_id = data.get("session_id")
    test_size = float(data.get("test_size", 0.2))

    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    processed = SESSIONS[session_id]["processed"]
    if processed is None:
        return jsonify({"error": "Preprocessing not done"}), 400

    X, y = processed

    if X.shape[1] == 0:
        return jsonify({"error": "Dataset has no usable features"}), 400

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    except Exception as e:
        return jsonify({"error": "Split failed", "details": str(e)}), 400

    SESSIONS[session_id]["train_data"] = (X_train, X_test, y_train, y_test)

    return jsonify({
        "message": "Split completed",
        "X_train_rows": len(X_train),
        "X_test_rows": len(X_test)
    })


# ------------------------------------------------------
# 4. TRAIN MODEL ROUTE
# ------------------------------------------------------
@app.route("/train", methods=["POST"])
def train():
    data = request.json
    session_id = data.get("session_id")
    model_type = data.get("model")

    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    train_data = SESSIONS[session_id]["train_data"]
    if train_data is None:
        return jsonify({"error": "Split not done"}), 400

    X_train, X_test, y_train, y_test = train_data

    # Select model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=2000)
    else:
        model = DecisionTreeClassifier()

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
    except Exception as e:
        return jsonify({"error": "Training failed", "details": str(e)}), 400

    model_path = os.path.join(MODEL_DIR, f"{session_id}_{model_type}.joblib")
    joblib.dump(model, model_path)

    SESSIONS[session_id]["metrics"] = {
        "accuracy": acc,
        "report": report
    }

    return jsonify({"message": "Training complete", "accuracy": acc})


# ------------------------------------------------------
# 5. RESULTS ROUTE
# ------------------------------------------------------
@app.route("/results", methods=["GET"])
def results():
    session_id = request.args.get("session_id")

    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    metrics = SESSIONS[session_id]["metrics"]
    if metrics is None:
        return jsonify({"error": "No results available"}), 400

    return jsonify(metrics)


# ------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
