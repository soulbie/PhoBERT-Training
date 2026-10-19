"""
Các thành phần dùng chung cho toàn bộ dự án PhoBERT + GCN:
  - Cấu hình chung (tên mô hình gốc, nhãn, độ dài tối đa, đường dẫn)
  - Kiến trúc mô hình GCNLayer / PhoBertGCN
  - Khởi tạo VnCoreNLP (tự tải về, xử lý đường dẫn có dấu cách, kiểm tra Java)
  - Hàm tiền xử lý văn bản dùng chung cho train / evaluate / predict
"""
import os
import shutil
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

# ==========================================
# CẤU HÌNH CHUNG
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODEL_DIR = ROOT_DIR / "phobert_gcn_model"
BEST_MODEL_DIR = MODEL_DIR / "best_model"

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# KIẾN TRÚC MÔ HÌNH
# ==========================================
class GCNLayer(nn.Module):
    """Một lớp tích chập đồ thị (Graph Convolution): H' = A · (H W) + b"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        # x: (batch, seq_len, in_features), adj: (batch, seq_len, seq_len)
        support = self.linear(x)        # (batch, seq_len, out_features)
        out = torch.bmm(adj, support)   # (batch, seq_len, out_features)
        return out + self.bias


class PhoBertGCN(nn.Module):
    """PhoBERT trích xuất đặc trưng node; ma trận Self-Attention layer cuối làm ma trận kề cho 2 lớp GCN."""

    def __init__(self, model_name=MODEL_NAME, num_labels=3, pretrained_backbone=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        if pretrained_backbone:
            self.phobert = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            # Khi sẽ nạp trọng số đã fine-tune ngay sau đó thì không cần tải ~540MB trọng số PhoBERT gốc
            self.phobert = AutoModel.from_config(self.config, attn_implementation="eager")

        hidden_size = self.config.hidden_size
        self.gcn1 = GCNLayer(hidden_size, 256)
        self.gcn2 = GCNLayer(256, 128)
        self.classifier = nn.Linear(128, num_labels)
        self.num_labels = num_labels

    @classmethod
    def from_trained(cls, model_dir=BEST_MODEL_DIR, device=None):
        """Nạp mô hình đã huấn luyện từ thư mục chứa model.safetensors, chuyển sang chế độ eval."""
        from safetensors.torch import load_model

        weights = Path(model_dir) / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(
                f"Không tìm thấy trọng số: {weights}\n"
                "Hãy chạy `python train_gcn.py` hoặc đặt thư mục best_model đã huấn luyện vào đúng vị trí."
            )
        model = cls(MODEL_NAME, num_labels=len(id2label), pretrained_backbone=False)
        load_model(model, str(weights), device="cpu")
        return model.to(device or get_device()).eval()

    def forward(self, input_ids, attention_mask, labels=None, return_graph=False, **kwargs):
        # 1. Trích xuất đặc trưng và lấy ma trận Self-Attention
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        H = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # 2. Ma trận kề = attention của layer cuối, trung bình các head (Softmax đã chuẩn hóa mỗi hàng có tổng = 1)
        A_norm = outputs.attentions[-1].mean(dim=1)  # (batch, seq_len, seq_len)

        # 3. Truyền đặc trưng node (H) và cạnh (A_norm) qua 2 lớp GCN
        x = F.relu(self.gcn1(H, A_norm))
        x = F.relu(self.gcn2(x, A_norm))  # (batch, seq_len, 128)

        # 4. Global Mean Pooling trên các token thật (bỏ padding) -> Graph Embedding
        mask = attention_mask.unsqueeze(-1).float()
        pooled_x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (batch, 128)

        # 5. Phân loại
        logits = self.classifier(pooled_x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        # Chỉ trả về ma trận kề khi cần trực quan hóa, để Trainer không gom nó vào predictions
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=(A_norm,) if return_graph else None,
        )


# ==========================================
# VNCORENLP (TÁCH TỪ TIẾNG VIỆT)
# ==========================================
VNCORENLP_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
VNCORENLP_FILES = [
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
    "models/postagger/vi-tagger",
    "models/ner/vi-500brownclusters.xz",
    "models/ner/vi-ner.xz",
    "models/ner/vi-pretrainedembeddings.xz",
    "models/dep/vi-dep.xz",
]


def _vncorenlp_dir():
    """Thư mục chứa VnCoreNLP. Ưu tiên biến môi trường VNCORENLP_DIR, sau đó ./vncorenlp trong dự án.
    VnCoreNLP (Java) đổi dấu cách thành %20 nên không tìm thấy model -> khi đó dùng ~/.cache/vncorenlp."""
    if os.environ.get("VNCORENLP_DIR"):
        return os.path.abspath(os.environ["VNCORENLP_DIR"])
    for candidate in (ROOT_DIR / "vncorenlp", Path.home() / ".cache" / "vncorenlp"):
        if " " not in str(candidate):
            return str(candidate)
    raise RuntimeError(
        "Đường dẫn dự án và thư mục home đều chứa dấu cách, VnCoreNLP không chạy được.\n"
        "Hãy đặt biến môi trường VNCORENLP_DIR tới một thư mục không có dấu cách, ví dụ: C:\\vncorenlp"
    )


def download_vncorenlp(save_dir):
    """Tải VnCoreNLP bằng urllib (py_vncorenlp.download_model cần wget, Windows không có sẵn)."""
    import urllib.request

    for rel in VNCORENLP_FILES:
        dst = os.path.join(save_dir, rel)
        if os.path.exists(dst):
            continue
        print(f"  - Đang tải {rel}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urllib.request.urlretrieve(VNCORENLP_URL + rel, dst + ".part")
        os.replace(dst + ".part", dst)


def load_segmenter(annotators=("wseg",)):
    """Khởi tạo VnCoreNLP (tự tải về ở lần chạy đầu). Cần Java 8+ (JDK) để chạy."""
    if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        raise RuntimeError(
            "VnCoreNLP cần Java (JDK 8 trở lên). Hãy cài JDK (ví dụ https://adoptium.net) "
            "và đặt biến môi trường JAVA_HOME, sau đó mở lại terminal."
        )
    import py_vncorenlp

    vncorenlp_dir = _vncorenlp_dir()
    if not all(os.path.exists(os.path.join(vncorenlp_dir, f)) for f in VNCORENLP_FILES):
        print(f"[INFO] Chưa có đủ VnCoreNLP, đang tải về {vncorenlp_dir} ...")
        download_vncorenlp(vncorenlp_dir)

    # py_vncorenlp đổi CWD sang thư mục VnCoreNLP, nên phải trả lại CWD gốc
    original_cwd = os.getcwd()
    try:
        segmenter = py_vncorenlp.VnCoreNLP(annotators=list(annotators), save_dir=vncorenlp_dir)
    finally:
        os.chdir(original_cwd)
    return segmenter


def segment_text(segmenter, text):
    """Chuẩn hóa Unicode NFC rồi tách từ (PhoBERT yêu cầu đầu vào đã tách từ, vd: 'giảng_viên')."""
    text = unicodedata.normalize("NFC", str(text).strip())
    return " ".join(segmenter.word_segment(text))
