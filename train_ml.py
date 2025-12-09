import os
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

from utils_preprocess import clean_en, clean_vi, clean_mix


# ============================================================
#                TẠO THƯ MỤC LƯU MODEL
# ============================================================
MODEL_DIR = "models/best"
TFIDF_DIR = "models/tfidf"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TFIDF_DIR, exist_ok=True)


# ============================================================
#      HÀM TIỀN XỬ LÝ DỮ LIỆU CHO EN – VI – MIX
# ============================================================
def preprocess_dataset(df, lang):
    """
    Xử lý text theo từng loại ngôn ngữ.
    """

    if lang == "EN":
        df["clean"] = df["text"].apply(clean_en)

    elif lang == "VI":
        df["clean"] = df["post_message"].apply(clean_vi)

    elif lang == "MIX":
        df["clean"] = df["clean"].apply(clean_mix)

    # Xóa dữ liệu rỗng
    df["clean"] = df["clean"].astype(str).str.strip()
    df = df[df["clean"] != ""]

    return df



# ============================================================
#            5-FOLD CV & LƯU BEST MODEL
# ============================================================
def train_5_fold(texts, labels, lang="EN"):
    """
    Train LogisticRegression, NaiveBayes, SVM và chọn model tốt nhất.
    Lưu best model + TF-IDF tương ứng.
    """

    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000),
        "NaiveBayes": MultinomialNB(),
        "SVM": LinearSVC()
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    X = texts
    y = labels

    fold = 1
    for train_idx, test_idx in kf.split(X):

        print(f"\n===== {lang} | Fold {fold} =====")
        fold += 1

        # Tách dữ liệu của fold
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2))
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        # SMOTE cân bằng
        if len(set(y_train)) > 1:
            X_res, y_res = SMOTE().fit_resample(X_train_vec, y_train)
        else:
            X_res, y_res = X_train_vec, y_train

        fold_metrics = {}

        # Train từng model
        for mname, clf in models.items():
            clf.fit(X_res, y_res)
            pred = clf.predict(X_test_vec)

            acc = accuracy_score(y_test, pred)
            pre = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)

            fold_metrics[mname] = (acc, pre, rec, f1)

            print(f"{mname}: ACC={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}")

        results.append(fold_metrics)

    # ============================================================
    #                TÍNH TRUNG BÌNH SCORE
    # ============================================================
    best_model_name = None
    best_f1 = -1

    for mname in models.keys():

        acc = np.mean([r[mname][0] for r in results])
        pre = np.mean([r[mname][1] for r in results])
        rec = np.mean([r[mname][2] for r in results])
        f1 = np.mean([r[mname][3] for r in results])

        print(f"\n>>> {lang} AVG {mname}: ACC={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}")

        # Chọn model có F1 cao nhất
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = mname

    print(f"\n🎯 BEST MODEL for {lang} = {best_model_name} | F1={best_f1:.4f}\n")

    # ============================================================
    #             TRAIN FULL DATA & LƯU BEST MODEL
    # ============================================================
    tfidf_full = TfidfVectorizer(max_features=12000, ngram_range=(1, 2))
    X_full = tfidf_full.fit_transform(X)

    if len(set(y)) > 1:
        X_bal, y_bal = SMOTE().fit_resample(X_full, y)
    else:
        X_bal, y_bal = X_full, y

    best_model = models[best_model_name]
    best_model.fit(X_bal, y_bal)

    # Lưu file
    joblib.dump(best_model, f"{MODEL_DIR}/best_{lang}.pkl")
    joblib.dump(tfidf_full, f"{TFIDF_DIR}/tfidf_{lang}.pkl")

    print(f"📌 Saved model: models/best/best_{lang}.pkl")
    print(f"📌 Saved TF-IDF: models/tfidf/tfidf_{lang}.pkl")

    return best_model_name, best_f1



# ============================================================
#                    CHẠY TRAIN 3 DỮ LIỆU
# ============================================================
print("\n====== TRAINING STARTED ======\n")

# ---------- TIẾNG ANH ----------
df_en = pd.read_csv("train.csv").dropna()
df_en = preprocess_dataset(df_en, "EN")
train_5_fold(df_en["clean"].tolist(), df_en["label"].tolist(), "EN")

# ---------- TIẾNG VIỆT ----------
df_vi = pd.read_csv("public_train.csv").dropna()
df_vi = preprocess_dataset(df_vi, "VI")
train_5_fold(df_vi["clean"].tolist(), df_vi["label"].tolist(), "VI")

# ---------- MIX (ghép 2 file) ----------
df_mix = pd.concat([
    pd.DataFrame({"clean": df_en["clean"], "label": df_en["label"]}),
    pd.DataFrame({"clean": df_vi["clean"], "label": df_vi["label"]})
]).sample(frac=1, random_state=42)

train_5_fold(df_mix["clean"].tolist(), df_mix["label"].tolist(), "MIX")

print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")