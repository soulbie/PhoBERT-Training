# PhoBERT GCN Sentiment Analysis

Dự án này sử dụng mô hình ngôn ngữ PhoBERT kết hợp với mạng Graph Convolutional Network (GCN) để phân loại cảm xúc văn bản tiếng Việt.

## 1. Cài đặt Môi trường (Installation)

Tạo môi trường ảo:
```bash
python -m venv venv

# Trên Windows:
venv\Scripts\activate
# Trên Mac/Linux:
source venv/bin/activate
```

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 2. Huấn luyện Mô hình (Training)

Sử dụng lệnh dưới đây để bắt đầu quá trình huấn luyện mô hình.
*Ghi chú: Trong lần chạy đầu tiên, mã nguồn sẽ tự động tải bộ công cụ tách từ VnCoreNLP về thư mục `./vncorenlp` của dự án.*

```bash
python train_gcn.py
```

Sau khi quá trình huấn luyện hoàn tất, hệ thống sẽ tự động xuất ra:
- Thư mục `phobert_gcn_model/best_model` chứa các trọng số (weights) đã được huấn luyện.
- File `metrics_YYYYMMDD_HHMMSS.txt` chứa kết quả đánh giá (Accuracy, Precision, Recall, F1-Score).
- File `loss_curve_gcn_YYYYMMDD_HHMMSS.png` mô tả biểu đồ độ lệch (Loss Curve).

## 3. Kiểm thử Mô hình (Testing)

Để kiểm thử mô hình trực tiếp qua giao diện dòng lệnh, sử dụng lệnh:

```bash
python test_gcn.py
```
*(Yêu cầu: Cần thực thi `train_gcn.py` trước để sinh ra thư mục `phobert_gcn_model/best_model`, hoặc tải file trọng số đã huấn luyện và đặt vào đúng vị trí).*

## 4. Trực quan hóa Đồ thị (Visualization)

Để trực quan hóa cấu trúc đồ thị Self-Attention do PhoBERT tạo ra trong quá trình tính toán, sử dụng lệnh:
```bash
python visualize_graph.py
```
