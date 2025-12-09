"""
utils_preprocess.py
File chứa hàm tiền xử lý tiếng Việt + tiếng Anh
Dùng cho cả SVM, CNN, BiLSTM, DistilBERT
"""

import re
import unicodedata
from underthesea import word_tokenize

# ---------------------------
# HÀM CHUẨN HÓA TIẾNG VIỆT
# ---------------------------

def normalize_vi(text):
    """
    Chuẩn hóa Unicode và dấu tiếng Việt.
    """
    text = unicodedata.normalize("NFC", text)
    return text


def clean_vi(text):
    """
    Tiền xử lý văn bản tiếng Việt:
    - chuẩn hóa unicode
    - bỏ URL, email, số
    - bỏ emoji
    - tách từ bằng underthesea
    - giữ lại chữ cái tiếng Việt
    """
    if not isinstance(text, str):
        text = str(text)

    text = normalize_vi(text)

    # Xóa URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Xóa email
    text = re.sub(r"\S+@\S+", " ", text)

    # Xóa emoji
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F" 
        "\U0001F300-\U0001F5FF" 
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    # Xóa ký tự không cần thiết
    text = re.sub(r"[^0-9a-zA-ZÀ-Ỵà-ỵ\s]", " ", text)

    # Tách từ
    text = word_tokenize(text, format="text")

    # Bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------
# HÀM TIỀN XỬ LÝ TIẾNG ANH
# ---------------------------

def clean_en(text):
    """
    Tiền xử lý văn bản tiếng Anh:
    - lowercase
    - bỏ URL/email
    - bỏ kí tự đặc biệt
    - giữ lại chữ cái và số
    """
    if not isinstance(text, str):
        text = str(text)

    # lowercase
    text = text.lower()

    # Xóa URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Xóa email
    text = re.sub(r"\S+@\S+", " ", text)

    # Xóa kí tự không cần thiết
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------
# HÀM TIỀN XỬ LÝ MIX (EN + VI)
# ---------------------------

def clean_mix(text):
    """
    Dùng chung logic của tiếng Anh + tiếng Việt.
    Thích hợp cho mô hình MIX.
    """
    # normalize unicode tiếng Việt
    text = normalize_vi(text)

    # Xóa URL/email
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Xóa emoji
    text = re.sub(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        " ",
        text,
    )

    # Giữ lại chữ cái tiếng Anh + tiếng Việt + số
    text = re.sub(r"[^0-9a-zA-ZÀ-Ỵà-ỵ\s]", " ", text)

    # Bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text