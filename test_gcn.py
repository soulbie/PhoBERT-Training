import sys
import unicodedata
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import py_vncorenlp
from transformers import AutoTokenizer, AutoConfig, AutoModel
from safetensors.torch import load_model

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# 1. Định nghĩa lại Kiến trúc GCN
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.bmm(adj, support)
        return out + self.bias

class PhoBertGCN(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        hidden_size = self.config.hidden_size
        self.gcn1 = GCNLayer(hidden_size, 256)
        self.gcn2 = GCNLayer(256, 128)
        self.classifier = nn.Linear(128, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        H = outputs.last_hidden_state
        last_layer_attn = outputs.attentions[-1]
        A_norm = last_layer_attn.mean(dim=1)
        
        x = self.gcn1(H, A_norm)
        x = F.relu(x)
        x = self.gcn2(x, A_norm)
        x = F.relu(x)
        
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_x = sum_embeddings / sum_mask
        
        logits = self.classifier(pooled_x)
        return logits

# ==========================================
# 2. Khởi tạo & Load Trọng số
# ==========================================
print("[INFO] Đang tải công cụ tách từ VnCoreNLP...")
import os
vncorenlp_dir = os.path.abspath("./vncorenlp")
if not os.path.exists(vncorenlp_dir):
    print("Chưa có VnCoreNLP, đang tiến hành tải xuống...")
    os.makedirs(vncorenlp_dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=vncorenlp_dir)

original_cwd = os.getcwd()
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
os.chdir(original_cwd)

print("[INFO] Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

print("[INFO] Đang xây dựng mạng PhoBERT + GCN và nạp trọng số đã huấn luyện...")
model = PhoBertGCN("vinai/phobert-base", num_labels=3)

# Nạp file trọng số Safetensors từ thư mục bạn đã copy lại
load_model(model, "phobert_gcn_model/best_model/model.safetensors")
model.eval() # Chuyển sang chế độ test (chỉ dự đoán)

# Mapping nhãn
id2label = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}

print("\n[INFO] KHỞI TẠO HOÀN TẤT! Gõ 'exit' hoặc 'quit' để dừng chương trình.\n")

# ==========================================
# 3. Vòng lặp Kiểm thử (Interactive Test)
# ==========================================
while True:
    try:
        cau_test = input("> Text: ")
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Đã thoát chương trình.")
        break
        
    if cau_test.strip().lower() in ['thoát', 'exit', 'quit', 'q']:
        print("[INFO] Đã thoát chương trình.")
        break
        
    if not cau_test.strip():
        continue

    # Chuẩn hóa Unicode NFC
    cau_test = unicodedata.normalize('NFC', cau_test)

    # Tiền xử lý tách từ
    cau_da_tach = " ".join(rdrsegmenter.word_segment(cau_test))
    
    # Mã hóa thành Tensor
    inputs = tokenizer(cau_da_tach, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    # Dự đoán (Tắt tính toán đạo hàm để tăng tốc độ)
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # Áp dụng Softmax để quy đổi Logits sang xác suất phần trăm
        probs = torch.softmax(logits, dim=1)
        
        # Lấy nhãn có xác suất cao nhất
        predicted_class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class_id].item() * 100
        
    nhan = id2label[predicted_class_id]
    print(f"  => Label: {nhan:<12} | Confidence: {confidence:.2f}%\n")
