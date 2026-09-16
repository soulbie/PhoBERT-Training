import os
import datetime
import pandas as pd
import numpy as np
import py_vncorenlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

# ==========================================
# 1. CÀI ĐẶT & KHỞI TẠO VNCORENLP
# ==========================================
print("--- Đang thiết lập VnCoreNLP ---")
# Sử dụng đường dẫn tương đối để dễ dàng đưa lên GitHub
vncorenlp_dir = os.path.abspath("./vncorenlp")
if not os.path.exists(vncorenlp_dir):
    print("Chưa có VnCoreNLP, đang tiến hành tải xuống...")
    os.makedirs(vncorenlp_dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=vncorenlp_dir)

# py_vncorenlp sẽ đổi CWD, nên ta phải lưu lại đường dẫn gốc
original_cwd = os.getcwd()
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
# Phục hồi CWD về thư mục dự án
os.chdir(original_cwd)

# ==========================================
# 2. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
print("--- Đang xử lý dữ liệu ---")
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, 'synthetic_train.csv')
val_path = os.path.join(current_dir, 'synthetic_val.csv')

try:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
except FileNotFoundError:
    print("Không tìm thấy file .csv, đang thử đuôi kép .csv.csv...")
    train_df = pd.read_csv(train_path + '.csv')
    val_df = pd.read_csv(val_path + '.csv')

label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
id2label = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}

train_df['label'] = train_df['sentiment'].map(label2id)
val_df['label'] = val_df['sentiment'].map(label2id)

def segment_text(text):
    sentences = rdrsegmenter.word_segment(text)
    return " ".join(sentences)

train_df['segmented_text'] = train_df['sentence'].apply(segment_text)
val_df['segmented_text'] = val_df['sentence'].apply(segment_text)

# ==========================================
# 3. CHUYỂN ĐỔI DATASET & TOKENIZATION
# ==========================================
print("--- Đang Tokenize dữ liệu ---")
train_dataset = Dataset.from_pandas(train_df[['segmented_text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['segmented_text', 'label']])

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def tokenize_function(examples):
    return tokenizer(examples["segmented_text"], padding="max_length", truncation=True, max_length=128)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)

# ==========================================
# 4. THIẾT KẾ MẠNG ĐỒ THỊ (GCN LAYER) VÀ CUSTOM MODEL
# ==========================================
print("--- Khởi tạo Mô hình PhoBERT + GCN ---")

class GCNLayer(nn.Module):
    """Một lớp tích chập đồ thị (Graph Convolution) thuần túy bằng PyTorch"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        # x: (batch, seq_len, in_features)
        # adj: (batch, seq_len, seq_len)
        support = self.linear(x) # (batch, seq_len, out_features)
        out = torch.bmm(adj, support) # (batch, seq_len, out_features)
        return out + self.bias

class PhoBertGCN(nn.Module):
    """Mô hình kết hợp PhoBERT làm trích xuất đặc trưng node, sau đó chạy qua GCN"""
    def __init__(self, model_name, num_labels, window_size=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        
        # Nếu máy yếu hoặc muốn train nhanh, có thể bỏ comment dòng dưới để Đóng băng (Freeze) PhoBERT
        # for param in self.phobert.parameters(): param.requires_grad = False
        
        hidden_size = self.config.hidden_size
        self.gcn1 = GCNLayer(hidden_size, 256)
        self.gcn2 = GCNLayer(256, 128)
        self.classifier = nn.Linear(128, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # 1. Trích xuất đặc trưng từ vựng và lấy ma trận Self-Attention
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        H = outputs.last_hidden_state # (batch, seq_len, hidden_size)
        
        # 2. Xây dựng đồ thị (Adjacency Matrix) dựa trên Self-Attention của PhoBERT
        # Lấy ma trận attention từ layer cuối cùng của PhoBERT
        last_layer_attn = outputs.attentions[-1] # (batch, num_heads, seq_len, seq_len)
        
        # Tính trung bình các heads để thu được ma trận kề tổng quát
        A_norm = last_layer_attn.mean(dim=1) # (batch, seq_len, seq_len)
        
        # Lưu ý: Ma trận Attention vốn đã được hàm Softmax chuẩn hóa (tổng mỗi hàng = 1), 
        # nên nó hoàn toàn thỏa mãn tính chất của Ma trận kề chuẩn hóa (Normalized Adjacency Matrix) cho GCN!
        
        # 3. Đưa Node Features (H) và Edges (A_norm) qua mạng GCN
        x = self.gcn1(H, A_norm)
        x = F.relu(x)
        
        x = self.gcn2(x, A_norm)
        x = F.relu(x) # (batch, seq_len, 128)
        
        # 4. Global Mean Pooling: Gom cụm toàn bộ Node lại thành 1 Graph Embedding
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_x = sum_embeddings / sum_mask # (batch, 128)
        
        # 5. Phân loại (Linear Classifier)
        logits = self.classifier(pooled_x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        # Trả về theo định dạng HuggingFace Trainer yêu cầu
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

# Khởi tạo mô hình
model = PhoBertGCN(model_name="vinai/phobert-base", num_labels=3)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# ==========================================
# 5. HUẤN LUYỆN (TRAINING)
# ==========================================
print("--- Bắt đầu Huấn luyện GCN ---")
training_args = TrainingArguments(
    output_dir="./phobert_gcn_model",       # Lưu ở thư mục mới
    eval_strategy="epoch",
    learning_rate=3e-5,                     # Tăng LR một chút vì GCN khởi tạo ngẫu nhiên
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",                  
    load_best_model_at_end=True,
    remove_unused_columns=False             # CẦN THIẾT: Trainer không được tự ý xóa cột đầu vào của custom model
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    compute_metrics=compute_metrics,
)

# Chạy huấn luyện
trainer.train()

# ==========================================
# 6. LƯU MÔ HÌNH VÀ VẼ BIỂU ĐỒ LOSS
# ==========================================
trainer.save_model("./phobert_gcn_model/best_model")
tokenizer.save_pretrained("./phobert_gcn_model/best_model")
print("--- Huấn luyện hoàn tất. Đã lưu model dạng Đồ thị (GCN) vào phobert_gcn_model ---")

# Vẽ biểu đồ Loss
try:
    log_history = trainer.state.log_history
    train_loss = [log['loss'] for log in log_history if 'loss' in log]
    train_steps = [log['step'] for log in log_history if 'loss' in log]
    
    val_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    val_steps = [log['step'] for log in log_history if 'eval_loss' in log]

    
    # ----------------------------------
    # TẠO DẤU THỜI GIAN (TIMESTAMP)
    # ----------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Vẽ biểu đồ Loss
    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(train_steps, train_loss, label='Training Loss', color='blue', linewidth=2)
    if val_loss:
        plt.plot(val_steps, val_loss, label='Validation Loss', color='red', linestyle='--', marker='o', linewidth=2)

    plt.title('Biểu đồ Lỗi (Loss Curve) - Mô hình PhoBERT GCN')
    plt.xlabel('Số vòng lặp (Steps)')
    plt.ylabel('Độ lệch (Cross-Entropy Loss)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    loss_img_name = f'loss_curve_gcn_{timestamp}.png'
    plt.savefig(loss_img_name, dpi=300)
    print(f"--- Đã xuất biểu đồ Loss thành công: {loss_img_name} ---")
    
    # Lấy các chỉ số cuối cùng (từ bước Eval cuối)
    final_eval_metrics = {}
    for log in reversed(log_history):
        if 'eval_loss' in log:
            final_eval_metrics = log
            break
            
    metrics_file_name = f'metrics_{timestamp}.txt'
    with open(metrics_file_name, 'w', encoding='utf-8') as f:
        f.write(f"=== KẾT QUẢ HUẤN LUYỆN GCN ({timestamp}) ===\n")
        for key, value in final_eval_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n=== LỊCH SỬ LOG (LOG HISTORY) ===\n")
        for log in log_history:
            f.write(f"{log}\n")
    print(f"--- Đã lưu đầy đủ chỉ số ra file: {metrics_file_name} ---")

except Exception as e:
    print(f"Không thể vẽ biểu đồ Loss hoặc lưu metrics: {e}")
