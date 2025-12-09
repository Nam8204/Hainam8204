// ==========================================================
// app.js
// Quản lý tương tác giao diện người dùng của hệ thống
// ==========================================================

document.addEventListener("DOMContentLoaded", function () {

    // --------------------------------------------------------
    // LẤY CÁC PHẦN TỬ TRÊN HTML
    // --------------------------------------------------------
    const form = document.getElementById("predict-form");
    const inputText = document.getElementById("input_text");
    const modelSelect = document.getElementById("model_name");

    const btnPredict = document.getElementById("btn-predict");
    const btnClear = document.getElementById("btn-clear");

    const resultCard = document.getElementById("result-card");

    // --------------------------------------------------------
    // XỬ LÝ NÚT "XÓA"
    // --------------------------------------------------------
    btnClear.addEventListener("click", () => {
        inputText.value = "";
        resultCard.className = "result-card empty";
        resultCard.innerHTML = `<p class="placeholder">Chưa có kết quả</p>`;
    });

    // --------------------------------------------------------
    // XỬ LÝ NÚT "VÍ DỤ"
    // --------------------------------------------------------
    document.querySelectorAll(".example-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            inputText.value = btn.dataset.text;
        });
    });

    // --------------------------------------------------------
    // GỬI DỮ LIỆU ĐI DỰ ĐOÁN
    // --------------------------------------------------------
    form.addEventListener("submit", async function (e) {
        e.preventDefault();

        const text = inputText.value.trim();
        const model = modelSelect.value;

        // Kiểm tra dữ liệu
        if (!text) {
            showError("Vui lòng nhập nội dung văn bản.");
            return;
        }
        if (!model) {
            showError("Vui lòng chọn mô hình.");
            return;
        }

        // Hiển thị trạng thái loading
        showLoading();

        // Tạo FormData để gửi đi
        const formData = new FormData();
        formData.append("input_text", text);
        formData.append("model_name", model);

        try {
            // Gửi POST request về Flask
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showError(data.error);
            } else {
                showResult(data);
            }

        } catch (err) {
            showError("Lỗi kết nối tới server.");
        }
    });

    // --------------------------------------------------------
    // HIỂN THỊ LOADING
    // --------------------------------------------------------
    function showLoading() {
        resultCard.className = "result-card loading";
        resultCard.innerHTML = `
            <div class="loader"></div>
            <p>Đang phân tích...</p>
        `;
    }

    // --------------------------------------------------------
    // HIỂN THỊ LỖI
    // --------------------------------------------------------
    function showError(msg) {
        resultCard.className = "result-card error";
        resultCard.innerHTML = `
            <p class="error-text">⚠️ ${msg}</p>
        `;
    }

    // --------------------------------------------------------
    // HIỂN THỊ KẾT QUẢ
    // --------------------------------------------------------
    function showResult(data) {

        // Xác định màu theo loại tin
        let cls = "result-card";
        if (data.raw === 1) cls += " fake";
        if (data.raw === 0) cls += " real";

        resultCard.className = cls;

        resultCard.innerHTML = `
            <p class="result-label">${data.label}</p>
            <p class="result-info">Mã nhãn: ${data.raw}</p>
            <p class="result-info">Model: ${data.model}</p>
            <p class="result-info">Thời gian: ${data.elapsed} ms</p>
            ${data.confidence !== undefined ?
                `<p class="result-info">Độ tin cậy: ${(data.confidence * 100).toFixed(1)}%</p>`
                : ""
            }
        `;
    }
});
