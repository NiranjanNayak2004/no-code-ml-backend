from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import uuid

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# =====================================================
# In-memory session store (demo-safe, predictable)
# =====================================================
SESSIONS = {}

# =====================================================
# 1. UPLOAD
# =====================================================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "df": df,
        "X": None,
        "y": None,
        "split": None,
        "accuracy": None
    }

    return jsonify({
        "session_id": session_id,
        "rows": df.shape[0],
        "columns": list(df.columns)
    })


# =====================================================
# 2. PREPROCESS
# =====================================================
@app.route("/preprocess", methods=["POST"])
def preprocess():
    data = request.json
    session_id = data["session_id"]
    method = data["method"]
    target = data["target"]

    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    df = SESSIONS[session_id]["df"]

    if target not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    y_raw = df[target]
    if y_raw.nunique() < 2:
        return jsonify({"error": "Target must have at least 2 classes"}), 400

    # Split X / y
    X = df.drop(columns=[target])
    y = y_raw.astype(str).astype("category").cat.codes

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Scale numeric columns only
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    if method == "standard":
        X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
    elif method == "minmax":
        X[numeric_cols] = MinMaxScaler().fit_transform(X[numeric_cols])

    SESSIONS[session_id]["X"] = X
    SESSIONS[session_id]["y"] = y

    return jsonify({
        "message": "Preprocessing successful",
        "feature_count": X.shape[1]
    })


# =====================================================
# 3. TRAIN–TEST SPLIT  ✅ FIXED
# =====================================================
@app.route("/split", methods=["POST"])
def split():
    data = request.json
    session_id = data["session_id"]
    test_size = float(data["test_size"])

    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    X = SESSIONS[session_id]["X"]
    y = SESSIONS[session_id]["y"]

    if X is None or y is None:
        return jsonify({"error": "Run preprocessing first"}), 400

    try:
        # ✅ TRY STRATIFIED SPLIT (BEST PRACTICE)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        stratified = True

    except ValueError:
        # 🔥 FALLBACK FOR SMALL / UNIQUE CLASSES
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )
        stratified = False

    SESSIONS[session_id]["split"] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    return jsonify({
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "stratified": stratified
    })



# =====================================================
# 4. TRAIN MODEL
# =====================================================
@app.route("/train", methods=["POST"])
def train():
    data = request.json
    session_id = data["session_id"]
    model_type = data["model"]

    split = SESSIONS.get(session_id, {}).get("split")
    if not split:
        return jsonify({"error": "Run train-test split first"}), 400

    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        return jsonify({"error": "Invalid model"}), 400

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    SESSIONS[session_id]["accuracy"] = acc

    return jsonify({
        "accuracy": round(acc, 3)
    })


# =====================================================
# 5. RESULTS
# =====================================================
@app.route("/results", methods=["GET"])
def results():
    session_id = request.args.get("session_id")
    acc = SESSIONS.get(session_id, {}).get("accuracy")

    return jsonify({
        "status": "done" if acc is not None else "not_trained",
        "accuracy": acc
    })


if __name__ == "__main__":
    app.run(debug=True)
