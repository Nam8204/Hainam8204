# ======================================================================
# app.py
# Flask Web API cho hệ thống phát hiện tin giả
# Hỗ trợ:
# - SVM (scikit-learn)
# - CNN / BiLSTM (Keras/TensorFlow)
# - DistilBERT (Transformers + Torch)
# ======================================================================

import os
import time
import pickle
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import logging

import numpy as np

# ------------------------------------------------------------
# Thử import TensorFlow (dành cho CNN / BiLSTM)
# ------------------------------------------------------------
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_OK = True
except:
    TF_OK = False
    load_model = None
    pad_sequences = None

# ------------------------------------------------------------
# Thử import Transformers (dành cho DistilBERT)
# ------------------------------------------------------------
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HF_OK = True
except:
    HF_OK = False
    torch = None

# ------------------------------------------------------------
# Import hàm preprocess_text
# ------------------------------------------------------------
try:
    from utils_preprocess import preprocess_text
except:
    def preprocess_text(x, lang="vi"):
        return x.lower().strip()


# ------------------------------------------------------------
# Khởi tạo Flask
# ------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.logger.setLevel(logging.INFO)

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"


# ======================================================================
# HÀM HỖ TRỢ TẢI PICKLE AN TOÀN
# ======================================================================
def load_pickle_safe(path):
    """Load file pickle, nếu lỗi thì trả None."""
    if not Path(path).exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None


# ======================================================================
# QUÉT MODEL TRONG THƯ MỤC VÀ LOAD TỰ ĐỘNG
# ======================================================================

# Cấu trúc thư mục:
# models/
#   svm/
#   tfidf/
#   cnn/
#   bilstm/
#   tokenizer/
#   distilbert_VI/
#   distilbert_EN/
#   distilbert_MIX/

LOADED = {}  # nơi lưu model đã load thành công

app.logger.info("------ BẮT ĐẦU LOAD MODEL ------")


# ======================================================================
# 1) LOAD SVM
# ======================================================================
def load_svm_models():
    svm_dir = MODELS_DIR / "svm"
    tfidf_dir = MODELS_DIR / "tfidf"

    app.logger.info(f"[LOAD] Thư mục SVM: {svm_dir.resolve()}")
    app.logger.info(f"[LOAD] Thư mục TF-IDF: {tfidf_dir.resolve()}")

    if not svm_dir.exists():
        app.logger.error("❌ Thư mục SVM không tồn tại!")
        return

    if not tfidf_dir.exists():
        app.logger.error("❌ Thư mục TF-IDF không tồn tại!")
        return

    # In danh sách file thật sự có trong thư mục TF-IDF
    app.logger.info(f"[DEBUG] Danh sách file TF-IDF có thật: {[f.name for f in tfidf_dir.glob('*')]}")
    
    # load tất cả file model trong thư mục svm
    for svm_file in svm_dir.glob("*.pkl"):
        name = svm_file.stem.upper()
        app.logger.info(f"[LOAD] Found SVM model file: {svm_file.name}")

        # suy ra ngôn ngữ
        if "EN" in name:
            lang = "EN"
        elif "VI" in name:
            lang = "VI"
        elif "MIX" in name:
            lang = "MIX"
        else:
            app.logger.warning(f"[SKIP] Không nhận diện được ngôn ngữ trong file: {svm_file.name}")
            continue

        tfidf_file = tfidf_dir / f"tfidf_{lang}.pkl"
        app.logger.info(f"[CHECK] Cần TF-IDF: {tfidf_file.name}")

        if not tfidf_file.exists():
            app.logger.error(f"❌ Không tìm thấy TF-IDF cho {lang}: {tfidf_file.resolve()}")
            continue

        # load tf-idf
        tfidf = load_pickle_safe(tfidf_file)
        if not tfidf:
            app.logger.error(f"❌ Load TF-IDF thất bại: {tfidf_file.name}")
            continue

        # load svm model
        model = load_pickle_safe(svm_file)
        if not model:
            app.logger.error(f"❌ Load SVM thất bại: {svm_file.name}")
            continue

        LOADED[f"SVM_{lang}"] = {
            "model": model,
            "vectorizer": tfidf,
            "type": "svm"
        }

        app.logger.info(f"✔ ĐÃ LOAD SVM: SVM_{lang}")

# ======================================================================
# 2) LOAD CNN / BiLSTM
# ======================================================================
def load_keras_models():
    if not TF_OK:
        app.logger.warning("TensorFlow chưa cài → CNN/BiLSTM sẽ không hoạt động")
        return

    for arch in ["cnn", "bilstm"]:
        model_dir = MODELS_DIR / arch
        token_dir = MODELS_DIR / "tokenizer"

        for h5_file in model_dir.glob("*.h5"):
            name = h5_file.stem.upper()

            lang = None
            if "VI" in name: lang = "VI"
            if "EN" in name: lang = "EN"
            if "MIX" in name: lang = "MIX"

            if lang is None:
                continue

            # tìm tokenizer
            tok_file = None
            for f in token_dir.glob("*.pkl"):
                if lang in f.stem.upper() and arch.upper() in f.stem.upper():
                    tok_file = f

            if not tok_file:
                app.logger.warning(f"{arch.upper()} thiếu tokenizer cho {lang}")
                continue

            model = load_model(h5_file)
            tok = load_pickle_safe(tok_file)

            if model and tok:
                model_name = f"{arch.upper()} ({lang})"
                LOADED[model_name] = {
                    "type": arch,
                    "model": model,
                    "tokenizer": tok
                }
                app.logger.info(f"ĐÃ LOAD: {model_name}")


# ======================================================================
# 3) LOAD DISTILBERT
# ======================================================================
def load_distilbert():
    if not HF_OK:
        app.logger.warning("Transformers chưa cài → DistilBERT bỏ qua")
        return

    for folder in ["distilbert_VI", "distilbert_EN", "distilbert_MIX"]:
        fd = MODELS_DIR / folder
        if not fd.exists():
            continue

        # Cần 2 file quan trọng:
        # - config.json
        # - pytorch_model.bin
        if not (fd / "config.json").exists():
            app.logger.warning(f"{folder} thiếu config.json → bỏ qua")
            continue

        if not (fd / "pytorch_model.bin").exists():
            app.logger.warning(f"{folder} thiếu pytorch_model.bin → bỏ qua")
            continue

        tokenizer = AutoTokenizer.from_pretrained(fd)
        model = AutoModelForSequenceClassification.from_pretrained(fd)

        model.to("cpu").eval()

        model_name = f"DistilBERT ({folder[-2:].upper()})"
        LOADED[model_name] = {
            "type": "distilbert",
            "tokenizer": tokenizer,
            "model": model
        }
        app.logger.info(f"ĐÃ LOAD: {model_name}")


# ======================================================================
# GỌI HÀM LOAD TẤT CẢ MODEL
# ======================================================================
load_svm_models()
load_keras_models()
load_distilbert()

app.logger.info(f"TỔNG SỐ MODEL TẢI THÀNH CÔNG: {len(LOADED)}")


# ======================================================================
# HÀM DỰ ĐOÁN CHUNG
# ======================================================================
def do_predict(model_name, text):

    cfg = LOADED[model_name]
    t = cfg["type"]

    # 1) SVM
    if t == "svm":
        text_clean = preprocess_text(text)
        X = cfg["vectorizer"].transform([text_clean])
        pred = cfg["model"].predict(X)[0]
        return {"label": "Tin giả" if pred == 1 else "Tin thật", "raw": int(pred)}

    # 2) CNN / BiLSTM
    if t in ["cnn", "bilstm"]:
        seq = cfg["tokenizer"].texts_to_sequences([preprocess_text(text)])
        seq = pad_sequences(seq, maxlen=128)
        prob = cfg["model"].predict(seq)[0][0]
        pred = 1 if prob >= 0.5 else 0
        return {"label": "Tin giả" if pred else "Tin thật", "raw": pred, "confidence": float(prob)}

    # 3) DistilBERT
    if t == "distilbert":
        tok = cfg["tokenizer"]
        model = cfg["model"]

        inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits
        prob = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(prob))

        return {"label": "Tin giả" if pred else "Tin thật", "raw": pred,
                "confidence": float(prob[0][pred])}


# ======================================================================
# ROUTES
# ======================================================================

@app.route("/")
def home():
    return render_template("index.html", models=sorted(LOADED.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("input_text", "")
    model_name = request.form.get("model_name", "")

    if model_name not in LOADED:
        return jsonify({"error": "Model không khả dụng"}), 400

    start = time.time()
    result = do_predict(model_name, text)
    result["model"] = model_name
    result["elapsed"] = int((time.time() - start) * 1000)

    return jsonify(result)

@app.route("/models")
def list_models():
    return jsonify({"models": sorted(LOADED.keys())})


# ======================================================================
# RUN SERVER
# ======================================================================
if __name__ == "__main__":
    app.run(debug=True)