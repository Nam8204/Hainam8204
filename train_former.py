import pandas as pd
import numpy as np
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Tắt các cảnh báo không cần thiết để màn hình gọn gàng
warnings.filterwarnings('ignore')

from utils_preprocess import clean_en, clean_vi

# ===== PHẦN KIỂM TRA THIẾT BỊ =====
# Hàm này kiểm tra xem máy tính có Intel XPU (GPU) hay không
# Nếu có thì sử dụng, không thì dùng CPU
def get_device():
    """
    Kiểm tra và chọn device phù hợp (XPU hoặc CPU)
    - Ưu tiên: Intel XPU > CPU
    - Trả về: device object và flag có XPU hay không
    """
    # Thử import Intel Extension for PyTorch
    try:
        import intel_extension_for_pytorch as ipex
        # Kiểm tra xem có Intel XPU (GPU) không
        if torch.xpu.is_available():
            device = torch.device("xpu")
            print(f" Using device: Intel XPU")
            print(f"   XPU Device: {torch.xpu.get_device_name(0)}")
            return device, True
    except:
        pass
    
    # Nếu không có XPU thì dùng CPU
    device = torch.device("cpu")
    print(f" Using device: CPU")
    print(f"     Intel XPU not available")
    return device, False

# Gọi hàm để xác định device sẽ dùng
device, has_xpu = get_device()

# ===== PHẦN 1: TẢI DỮ LIỆU =====
# Đọc 2 file CSV: train.csv (tiếng Anh) và public_train.csv (tiếng Việt)
# .dropna() để xóa các hàng có giá trị bị thiếu (NaN)
print("\n Loading data...")
df_en = pd.read_csv("train.csv").dropna()
df_vi = pd.read_csv("public_train.csv").dropna()

# ===== PHẦN 2: LÀM SẠCH DỮ LIỆU =====
# Áp dụng các hàm tiền xử lý văn bản để làm sạch dữ liệu
print(" Cleaning data...")

# Làm sạch dữ liệu tiếng Anh
# clean_en() trả về chuỗi đã được làm sạch
df_en["clean"] = df_en["text"].apply(clean_en)

# Làm sạch dữ liệu tiếng Việt
# clean_vi() trả về list các token, cần join thành chuỗi
df_vi["clean"] = df_vi["post_message"].apply(clean_vi).apply(lambda x: " ".join(x))

# Xóa các hàng mà văn bản sau khi làm sạch bị rỗng
# Điều này đảm bảo không có dữ liệu không hợp lệ
df_en = df_en[df_en["clean"].str.strip() != ""]
df_vi = df_vi[df_vi["clean"].str.strip() != ""]

# In ra số lượng mẫu sau khi làm sạch
print(f"   EN samples: {len(df_en)}")
print(f"   VI samples: {len(df_vi)}")

# ===== PHẦN 3: KẾT HỢP DỮ LIỆU =====
# Tạo một DataFrame kết hợp từ dữ liệu tiếng Anh và tiếng Việt
# sample(frac=1) để xáo trộn dữ liệu, random_state=42 để tái tạo được kết quả
df_mix = pd.concat([
    pd.DataFrame({"clean": df_en["clean"], "label": df_en["label"]}),
    pd.DataFrame({"clean": df_vi["clean"], "label": df_vi["label"]})
]).sample(frac=1, random_state=42)

print(f"   MIX samples: {len(df_mix)}")

# ===== PHẦN 4: ĐỊNH NGHĨA DATASET TÙY CHỈNH =====
# Class này chuyển đổi dữ liệu thành format mà PyTorch có thể sử dụng
class TextDataset(Dataset):
    """
    Custom Dataset cho văn bản đã được tokenize
    - encodings: dict chứa input_ids, attention_mask từ tokenizer
    - labels: nhãn tương ứng với mỗi văn bản
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Lưu encodings (output của tokenizer)
        self.labels = labels        # Lưu labels (nhãn)

    def __getitem__(self, idx):
        """
        Lấy một mẫu tại index idx
        - Chuyển tất cả encodings và label thành tensor
        - Trả về dict có cả input và label
        """
        # Chuyển mỗi phần tử trong encodings thành tensor
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Thêm label vào item
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Trả về tổng số mẫu trong dataset"""
        return len(self.labels)

# ===== PHẦN 5: HÀM HUẤN LUYỆN KHÔNG K-FOLD =====
def train_modern_ml(texts, labels, name="EN", model_name="distilbert-base-multilingual-cased"):
    """
    Hàm huấn luyện model KHÔNG sử dụng K-Fold Cross Validation
    Chỉ train 1 lần duy nhất với split 80% train / 20% validation
    
    Tham số:
    - texts: list các văn bản đã làm sạch
    - labels: list các nhãn tương ứng (0 hoặc 1)
    - name: tên của dataset (EN, VI, hoặc MIX)
    - model_name: tên model từ Hugging Face (mặc định: DistilBERT)
    
    DistilBERT là phiên bản nhẹ hơn của BERT:
    - Nhanh hơn 60%
    - Nhẹ hơn 40% (66M parameters thay vì 110M)
    - Giữ được 97% hiệu suất của BERT
    - Rất phù hợp cho máy yếu như Intel UHD 620
    """
    print(f"\n{'='*80}")
    print(f" TRAINING MODEL: {name} - Using {model_name}")
    print(f"{'='*80}")

    # Chuyển texts thành list và labels thành numpy array để dễ xử lý
    X_texts = texts
    y = np.array(labels)

    # ===== CHIA DỮ LIỆU TRAIN/VALIDATION =====
    # Chia 80% để train, 20% để validation
    # stratify=y để đảm bảo tỷ lệ các class giống nhau ở train và val
    # random_state=42 để kết quả có thể tái tạo
    print("\n Splitting data: 80% train / 20% validation...")
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        X_texts, y, 
        test_size=0.2,      # 20% cho validation
        random_state=42,    # Seed để tái tạo kết quả
        stratify=y          # Giữ tỷ lệ class cân bằng
    )
    
    print(f"   Train samples: {len(X_train_texts)}")
    print(f"   Val samples:   {len(X_val_texts)}")

    # ===== TOKENIZATION =====
    # Tokenizer chuyển văn bản thành số (token IDs) mà model hiểu được
    print("\n Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize dữ liệu train
    # truncation=True: cắt văn bản nếu dài quá max_length
    # padding=True: thêm padding để tất cả cùng độ dài
    # max_length=128: độ dài tối đa (phù hợp với DistilBERT)
    train_encodings = tokenizer(X_train_texts, truncation=True, padding=True, max_length=128)
    
    # Tokenize dữ liệu validation
    val_encodings = tokenizer(X_val_texts, truncation=True, padding=True, max_length=128)

    # ===== TẠO DATASET =====
    # Chuyển encodings và labels thành Dataset object
    train_dataset = TextDataset(train_encodings, y_train)
    val_dataset = TextDataset(val_encodings, y_val)

    # ===== LOAD MODEL =====
    # Load DistilBERT pretrained model
    # num_labels=2: vì bài toán phân loại nhị phân (0 hoặc 1)
    print("\n Loading DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # ===== TỐI ƯU MODEL CHO INTEL XPU =====
    # Intel Extension for PyTorch giúp tối ưu model cho Intel GPU
    if has_xpu:
        try:
            import intel_extension_for_pytorch as ipex
            model = model.to(device)  # Chuyển model sang XPU
            model = ipex.optimize(model)  # Tối ưu hóa cho Intel GPU
            print("    Model optimized for Intel XPU")
        except Exception as e:
            print(f"     XPU optimization failed: {e}")

    # ===== THIẾT LẬP TRAINING ARGUMENTS =====
    # Các tham số để điều khiển quá trình training
    training_args = TrainingArguments(
        output_dir=f"./results_{name}",             # Thư mục lưu kết quả
        num_train_epochs=3,                         # Số epoch (vòng lặp qua toàn bộ data)
        per_device_train_batch_size=16,             # Số mẫu xử lý cùng lúc khi train
        per_device_eval_batch_size=32,              # Batch size khi evaluate
        warmup_steps=500,                           # Số steps để learning rate tăng dần từ 0
        weight_decay=0.01,                          # L2 regularization để tránh overfitting
        logging_dir='./logs',                       # Thư mục lưu logs
        logging_steps=100,                          # Log metrics mỗi 100 steps
        eval_strategy="epoch",                      # Evaluate sau mỗi epoch
        save_strategy="epoch",                      # Lưu model sau mỗi epoch
        load_best_model_at_end=True,                # Load lại best model sau khi train xong
        metric_for_best_model="eval_loss",          # Dùng validation loss để chọn best model
        report_to="none",                           # Không gửi logs lên wandb/tensorboard
        use_cpu=not has_xpu,                        # Dùng CPU nếu không có XPU
        dataloader_pin_memory=False,                # Tắt pin memory (tránh warning)
        disable_tqdm=False,                         # Hiện progress bar
        no_cuda=True,                               # Tắt CUDA (vì dùng XPU/CPU)
    )

    # ===== TẠO TRAINER VÀ BẮT ĐẦU TRAINING =====
    # Trainer là class quản lý toàn bộ quá trình training
    print("\n  Training DistilBERT...")
    trainer = Trainer(
        model=model,                    # Model cần train
        args=training_args,             # Training arguments
        train_dataset=train_dataset,    # Dataset để train
        eval_dataset=val_dataset        # Dataset để evaluate
    )
    
    # Bắt đầu training - đây là bước tốn thời gian nhất
    trainer.train()

    # ===== DỰ ĐOÁN VÀ TÍNH METRICS TRÊN VALIDATION SET =====
    # Sau khi train xong, dùng model để dự đoán trên validation set
    print("\n Evaluating on validation set...")
    predictions = trainer.predict(val_dataset)
    
    # predictions.predictions là ma trận xác suất [n_samples, 2]
    # Mỗi hàng có 2 giá trị: xác suất của class 0 và class 1
    pred_prob = predictions.predictions
    
    # argmax để lấy class có xác suất cao nhất
    pred = np.argmax(pred_prob, axis=1)

    # Tính các metrics để đánh giá model
    # Accuracy: % dự đoán đúng
    acc = accuracy_score(y_val, pred)
    # Precision: trong các mẫu dự đoán positive, có bao nhiêu thực sự positive
    pre = precision_score(y_val, pred, zero_division=0)
    # Recall: trong các mẫu thực sự positive, model tìm được bao nhiêu
    rec = recall_score(y_val, pred, zero_division=0)
    # F1: trung bình điều hòa của Precision và Recall
    f1 = f1_score(y_val, pred, zero_division=0)

    # In kết quả đánh giá
    print(f"\n{'='*80}")
    print(f" VALIDATION RESULTS ({name}):")
    print(f"{'='*80}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {pre:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"{'='*80}\n")

    # ===== LƯU MODEL VÀ TOKENIZER =====
    # Lưu model đã train để có thể dùng lại sau này
    print(f" Saving model to distilbert_{name}/")
    model.save_pretrained(f"distilbert_{name}")      # Lưu model weights
    tokenizer.save_pretrained(f"distilbert_{name}")  # Lưu tokenizer config
    print(f" Model saved successfully!\n")

    # ===== GIẢI PHÓNG BỘ NHỚ =====
    # Xóa model và trainer để giải phóng RAM/VRAM
    del model, trainer
    if has_xpu:
        torch.xpu.empty_cache()  # Xóa cache của XPU

# ===== PHẦN 6: CHẠY HUẤN LUYỆN CHO TỪNG BỘ DỮ LIỆU =====
# Main function - điểm bắt đầu của chương trình
if __name__ == "__main__":
    print("\n" + "="*80)
    print("="*80)
    
    # Train model cho dữ liệu tiếng Anh
    # - Input: danh sách văn bản đã clean và nhãn tương ứng
    # - Output: model được lưu tại distilbert_EN/
    train_modern_ml(df_en["clean"].tolist(), df_en["label"].tolist(), "EN")
    
    # Train model cho dữ liệu tiếng Việt
    # - Input: danh sách văn bản đã clean và nhãn tương ứng
    # - Output: model được lưu tại distilbert_VI/
    train_modern_ml(df_vi["clean"].tolist(), df_vi["label"].tolist(), "VI")
    
    # Train model cho dữ liệu kết hợp (EN + VI)
    # - Input: danh sách văn bản đã clean và nhãn tương ứng
    # - Output: model được lưu tại distilbert_MIX/
    # - Model này có thể xử lý cả tiếng Anh và tiếng Việt
    train_modern_ml(df_mix["clean"].tolist(), df_mix["label"].tolist(), "MIX")
    
    print("\n" + "="*80)
    print(" ALL TRAINING COMPLETED!")
    print("="*80)
    print("\n Saved models:")
    print("   - distilbert_EN/   (English model)")
    print("   - distilbert_VI/   (Vietnamese model)")
    print("   - distilbert_MIX/  (Multilingual model)")
    print("\n You can now use these models for inference!")
    print("  Training time was much faster without K-Fold!")