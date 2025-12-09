import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Dense, Dropout

from utils_preprocess import clean_en, clean_vi

# ===== Phần 1: Tải dữ liệu =====
# - Đọc file CSV cho dữ liệu tiếng Anh và tiếng Việt.
# - Xóa các hàng có giá trị NaN để đảm bảo dữ liệu sạch.
df_en = pd.read_csv("train.csv").dropna()
df_vi = pd.read_csv("public_train.csv").dropna()

# ===== Phần 2: Làm sạch dữ liệu =====
# - Áp dụng hàm clean_en để làm sạch văn bản tiếng Anh (giả sử trả về chuỗi).
# - Áp dụng hàm clean_vi để làm sạch văn bản tiếng Việt (giả sử trả về list token), sau đó nối thành chuỗi.
df_en["clean"] = df_en["text"].apply(clean_en)
df_vi["clean"] = df_vi["post_message"].apply(clean_vi).apply(lambda x: " ".join(x))

# - Xóa các hàng mà văn bản sạch bị rỗng để tránh dữ liệu không hợp lệ.
df_en = df_en[df_en["clean"].str.strip() != ""]
df_vi = df_vi[df_vi["clean"].str.strip() != ""]

# ===== Phần 3: Kết hợp dữ liệu =====
# - Tạo DataFrame kết hợp từ dữ liệu tiếng Anh và tiếng Việt.
# - Xáo trộn dữ liệu với random_state để đảm bảo tính nhất quán.
df_mix = pd.concat([
    pd.DataFrame({"clean": df_en["clean"], "label": df_en["label"]}),
    pd.DataFrame({"clean": df_vi["clean"], "label": df_vi["label"]})
]).sample(frac=1, random_state=42)

# ===== Phần 4: Định nghĩa các mô hình deep learning =====
# - Xây dựng mô hình CNN cho phân loại văn bản.
# - input_dim là kích thước từ vựng (vocab size).
def build_cnn(input_dim):
    model = Sequential([
        Embedding(input_dim, 128),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# - Xây dựng mô hình BiLSTM cho phân loại văn bản.
# - input_dim là kích thước từ vựng (vocab size).
def build_bilstm(input_dim):
    model = Sequential([
        Embedding(input_dim, 128),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ===== Phần 5: Hàm huấn luyện với 5-fold cross-validation =====
# - Hàm này thực hiện K-Fold validation trên dữ liệu văn bản và nhãn.
# - Sử dụng Tokenizer và pad_sequences để chuyển văn bản thành sequences.
# - Huấn luyện CNN và BiLSTM, tính toán 4 chỉ số: Accuracy (ACC), Precision (P), Recall (R), F1-score (F1) cho từng fold và trung bình.
# - Tokenizer được fit riêng trên dữ liệu train của mỗi fold để tránh data leakage.
# - Cuối cùng huấn luyện mô hình đầy đủ và lưu model + tokenizer.
def train_dl(texts, labels, name="EN"):
    print(f"\n===== TRAINING DL MODELS ({name}) =====")

    X_texts = texts  # Danh sách văn bản sạch
    y = np.array(labels)  # Nhãn dưới dạng numpy array

    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Khởi tạo K-Fold với 5 splits

    for model_name in ["CNN", "BiLSTM"]:
        fold = 1  # Biến đếm fold
        acc_list, pre_list, rec_list, f1_list = [], [], [], []  # Lưu metrics của từng fold

        # Lặp qua từng fold
        for train_idx, test_idx in kf.split(X_texts):
            print(f"\n---- {model_name} Fold {fold} ----")

            # Tách dữ liệu train và test cho fold này (dùng texts gốc để tránh leakage)
            X_train_texts = [X_texts[i] for i in train_idx]
            X_test_texts = [X_texts[i] for i in test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Tokenize và pad sequences (fit tokenizer chỉ trên train)
            tokenizer = Tokenizer(num_words=20000)
            tokenizer.fit_on_texts(X_train_texts)
            train_seq = tokenizer.texts_to_sequences(X_train_texts)
            test_seq = tokenizer.texts_to_sequences(X_test_texts)
            X_train = pad_sequences(train_seq, maxlen=200)
            X_test = pad_sequences(test_seq, maxlen=200)

            # Xác định input_dim cho Embedding (num_words + 1)
            input_dim = tokenizer.num_words + 1

            # Xây dựng và huấn luyện mô hình
            if model_name == "CNN":
                model = build_cnn(input_dim)
            else:
                model = build_bilstm(input_dim)

            model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test), verbose=0)

            # Dự đoán và tính metrics
            pred_prob = model.predict(X_test)
            pred = (pred_prob > 0.5).astype(int).flatten()  # Chuyển sang nhãn binary

            acc = accuracy_score(y_test, pred)
            pre = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)

            # Lưu metrics vào list
            acc_list.append(acc)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)

            # In metrics cho fold này
            print(f"{model_name} Fold {fold}: ACC={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}")
            fold += 1

        # ===== Phần 6: Tính trung bình metrics qua các fold =====
        avg_acc = np.mean(acc_list)
        avg_pre = np.mean(pre_list)
        avg_rec = np.mean(rec_list)
        avg_f1 = np.mean(f1_list)

        # In trung bình metrics cho mô hình
        print(f">>> AVG {model_name} ({name}): ACC={avg_acc:.4f}, P={avg_pre:.4f}, R={avg_rec:.4f}, F1={avg_f1:.4f}")

        # ===== Phần 7: Huấn luyện mô hình đầy đủ trên toàn bộ dữ liệu =====
        # - Tokenize và pad toàn bộ dữ liệu.
        # - Huấn luyện mô hình và lưu file.
        tokenizer_full = Tokenizer(num_words=20000)
        tokenizer_full.fit_on_texts(X_texts)
        sequences_full = tokenizer_full.texts_to_sequences(X_texts)
        X_full = pad_sequences(sequences_full, maxlen=200)
        input_dim_full = tokenizer_full.num_words + 1

        final_model = build_cnn(input_dim_full) if model_name == "CNN" else build_bilstm(input_dim_full)
        final_model.fit(X_full, y, epochs=3, batch_size=64, verbose=0)

        final_model.save(f"{model_name.lower()}_{name}.h5")
        joblib.dump(tokenizer_full, f"tokenizer_{model_name.lower()}_{name}.pkl")

# ===== Phần 8: Chạy huấn luyện cho từng bộ dữ liệu =====
# - Gọi hàm train_dl cho dữ liệu tiếng Anh, tiếng Việt, và kết hợp.
train_dl(df_en["clean"].tolist(), df_en["label"].tolist(), "EN")
train_dl(df_vi["clean"].tolist(), df_vi["label"].tolist(), "VI")
train_dl(df_mix["clean"].tolist(), df_mix["label"].tolist(), "MIX")                                     