"""
Trực quan hóa Đồ thị Tri thức (Knowledge Graph) cho mô hình PhoBERT + GCN.

Cấu trúc đồ thị (tham khảo "Visualizing Knowledge Graphs" - Instructor):
  - Node / Edge / KnowledgeGraph là các Pydantic model, có update() (gộp & khử trùng lặp) và draw().
  - Các tầng:   Câu (cảm xúc dự đoán)  ->  Mệnh đề (cảm xúc riêng)  ->  Từ (cây phụ thuộc cú pháp)
  - Cạnh liền   : quan hệ cú pháp từ VnCoreNLP (chủ ngữ, bổ nghĩa, trạng ngữ...)
  - Cạnh nét đứt: liên kết Attention mạnh nhất mà GCN thực sự dùng làm ma trận kề (đã bỏ <s>, </s>)
  - Màu của từ  : mức đóng góp vào cảm xúc (xóa từ đó đi rồi đo P(tích cực) - P(tiêu cực) thay đổi bao nhiêu)
  - Danh từ (thực thể) được dùng chung giữa các câu, nên khi cập nhật lặp nhiều câu sẽ ra một đồ thị liên thông.

Cách dùng:
  python visualize_graph.py
  python visualize_graph.py "Câu thứ nhất." "Câu thứ hai." --top-k 2
"""
import os
import sys
import shutil
import logging
import argparse
import textwrap
import unicodedata
from typing import List, Optional

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Chỉ xuất file ảnh, không cần màn hình
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

from model import BEST_MODEL_DIR, MAX_LENGTH, MODEL_NAME, RESULTS_DIR, PhoBertGCN, get_device, id2label, load_segmenter

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_SENTENCES = [
    "Giảng viên này dạy rất tuyệt nhưng hay đến muộn.",
    "Giảng viên luôn nhiệt tình giải đáp thắc mắc của sinh viên.",
    "Phòng học quá nóng và máy chiếu thường xuyên bị hỏng.",
]

SENTIMENT_COLORS = {0: "#e74c3c", 1: "#95a5a6", 2: "#27ae60"}  # Tiêu cực / Trung tính / Tích cực
ENTITY_COLOR = "#f5c26b"
ATTN_COLOR = "#2e86de"
SYNTAX_COLOR = "#555555"

# Các từ nối dùng để tách mệnh đề, chúng trở thành nhãn cạnh giữa hai mệnh đề
CLAUSE_CONNECTIVES = {"nhưng", "nhưng_mà", "tuy_nhiên", "song", "còn", "mà", "tuy", "dù", "mặc_dù", "vì", "nên"}
CLAUSE_PUNCT = {",", ";", ":"}

# Nhãn quan hệ phụ thuộc của VnCoreNLP (VnDT) -> tiếng Việt
DEP_LABELS = {
    "sub": "chủ ngữ", "dob": "tân ngữ", "iob": "tân ngữ gián tiếp", "pob": "đối tượng giới từ",
    "nmod": "bổ nghĩa danh từ", "amod": "bổ nghĩa tính từ", "vmod": "bổ nghĩa động từ",
    "adv": "trạng ngữ", "det": "hạn định", "mnr": "cách thức", "tmp": "thời gian",
    "loc": "nơi chốn", "dir": "hướng", "prp": "mục đích", "prd": "vị ngữ",
    "coord": "kết hợp", "conj": "liên kết", "root": "gốc", "x": "khác", "dep": "phụ thuộc",
}


# ==========================================
# 1. CẤU TRÚC ĐỒ THỊ (Pydantic)
# ==========================================
class Node(BaseModel, frozen=True):
    id: str
    label: str
    color: str
    shape: str = "ellipse"
    level: int = 0           # 0 = câu, 1 = mệnh đề, 2 = từ


class Edge(BaseModel, frozen=True):
    source: str
    target: str
    label: str = ""
    color: str = "black"
    style: str = "solid"     # "solid" = cú pháp / cấu trúc, "dashed" = attention


class KnowledgeGraph(BaseModel):
    title: str = "Đồ thị tri thức PhoBERT + GCN"
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Gộp đồ thị khác vào đồ thị hiện tại, khử trùng lặp node theo id (giữ node xuất hiện trước)."""
        nodes = {n.id: n for n in self.nodes}
        for n in other.nodes:
            nodes.setdefault(n.id, n)
        edges = list(dict.fromkeys(self.edges + other.edges))
        return KnowledgeGraph(title=self.title, nodes=list(nodes.values()), edges=edges)

    def draw(self, prefix: str):
        """Xuất file .gv; render PNG bằng Graphviz nếu máy có, không thì dùng matplotlib."""
        dot = self._to_graphviz()
        if dot is not None:
            dot.save(prefix + ".gv")
            if shutil.which("dot"):
                dot.render(prefix, format="png", cleanup=True)
                return prefix + ".png"
        self._draw_matplotlib(prefix + ".png")
        return prefix + ".png"

    def _to_graphviz(self):
        try:
            from graphviz import Digraph
        except ImportError:
            return None
        dot = Digraph(comment=self.title)
        dot.attr(rankdir="TB", label=self.title, labelloc="t", fontsize="18", fontname="Arial")
        dot.attr("node", style="filled", fontname="Arial")
        dot.attr("edge", fontname="Arial", fontsize="10")
        for node in self.nodes:
            dot.node(node.id, node.label, fillcolor=node.color, shape=node.shape)
        for edge in self.edges:
            # Chỉ cạnh cấu trúc mới quyết định bố cục cây; cạnh attention / nối mệnh đề thì không
            dot.edge(edge.source, edge.target, label=edge.label, color=edge.color, fontcolor=edge.color,
                     style=edge.style, constraint="true" if edge.style == "solid" else "false")
        return dot

    def _tree_layout(self):
        """Bố cục cây gọn: lá xếp liên tiếp từ trái sang phải, node cha nằm giữa các con."""
        children = {n.id: [] for n in self.nodes}
        for e in self.edges:
            if e.style == "solid":
                children[e.source].append(e.target)
        widths = {n.id: max(1.0, 0.11 * max(len(line) for line in n.label.split("\n"))) for n in self.nodes}
        pos, slot = {}, [0.0]

        def place(node, depth):
            pos[node] = None
            kids = [c for c in children[node] if c not in pos]
            for c in kids:
                place(c, depth + 1)
            if kids:
                x = (pos[kids[0]][0] + pos[kids[-1]][0]) / 2
                # Node cha rộng hơn cả cây con thì chừa thêm chỗ bên phải
                slot[0] = max(slot[0], x + widths[node] / 2 + 0.3)
            else:
                x = slot[0] + widths[node] / 2
                slot[0] += widths[node] + 0.3
            pos[node] = (x, -depth)

        for root in [n.id for n in self.nodes if n.level == 0]:
            if root not in pos:
                place(root, 0)
                slot[0] += 1.0  # khoảng cách giữa các câu
        for n in self.nodes:
            if n.id not in pos:
                place(n.id, 0)
        return pos

    def _draw_matplotlib(self, path):
        pos = self._tree_layout()
        xs, ys = zip(*pos.values())
        fig, ax = plt.subplots(figsize=(max(12, 1.3 * (max(xs) - min(xs) + 2)), 3 + 1.7 * (max(ys) - min(ys) + 1)))

        for e in self.edges:
            attn = e.style == "dashed"
            ax.annotate("", xy=pos[e.target], xytext=pos[e.source],
                        arrowprops=dict(arrowstyle="-|>", color=e.color, lw=1.6 if attn or e.style == "bold" else 1.2,
                                        linestyle="--" if attn else "-", shrinkA=18, shrinkB=18,
                                        connectionstyle="arc3,rad=0.25" if attn else "arc3"))
            if e.label:
                (x1, y1), (x2, y2) = pos[e.source], pos[e.target]
                off = 0.12 if attn else 0.0
                ax.text((x1 + x2) / 2 + off, (y1 + y2) / 2 + off, e.label, fontsize=8, color=e.color,
                        ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))

        for n in self.nodes:
            boxstyle = "square,pad=0.5" if n.shape == "box" else "round,pad=0.5"
            ax.text(*pos[n.id], n.label, ha="center", va="center", zorder=3,
                    fontsize=10 if n.level == 2 else 11, fontweight="bold" if n.level < 2 else "normal",
                    bbox=dict(boxstyle=boxstyle, fc=n.color, ec="#333333", lw=1))

        legend = [
            Patch(fc=SENTIMENT_COLORS[2], ec="#333", label="Tích cực / từ kéo về tích cực"),
            Patch(fc=SENTIMENT_COLORS[0], ec="#333", label="Tiêu cực / từ kéo về tiêu cực"),
            Patch(fc=SENTIMENT_COLORS[1], ec="#333", label="Trung tính"),
            Patch(fc=ENTITY_COLOR, ec="#333", label="Thực thể (danh từ, dùng chung giữa các câu)"),
            Line2D([0], [0], color=SYNTAX_COLOR, lw=1.2, label="Quan hệ cú pháp / cấu trúc"),
            Line2D([0], [0], color=ATTN_COLOR, lw=1.2, label="Cú pháp trùng attention mạnh (nhãn ghi attn)"),
            Line2D([0], [0], color=ATTN_COLOR, lw=1.6, ls="--", label="Attention mạnh nhất (ma trận kề của GCN)"),
            Line2D([0], [0], color="#8e44ad", lw=1.6, label="Từ nối giữa hai mệnh đề"),
        ]
        ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        # ax.text / ax.annotate không tự co giãn trục nên phải đặt giới hạn theo vị trí node
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


# ==========================================
# 2. PHÂN TÍCH CÂU BẰNG MÔ HÌNH
# ==========================================
def blend(hex_a, hex_b, t):
    """Trộn màu từ hex_a (t=0) tới hex_b (t=1)."""
    a = np.array([int(hex_a[i:i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(hex_b[i:i + 2], 16) for i in (1, 3, 5)])
    c = (a + (b - a) * float(np.clip(t, 0, 1))).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def score_color(score):
    """score > 0: từ kéo câu về tích cực (xanh), score < 0: về tiêu cực (đỏ)."""
    base = "#f2f2f2"
    t = min(abs(score) / 0.5, 1.0)
    return blend(base, SENTIMENT_COLORS[2] if score > 0 else SENTIMENT_COLORS[0], t)


def display(word):
    return word.replace("_", " ")


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


class GraphBuilder:
    def __init__(self, model_dir, top_k=1, min_attn=0.1):
        print("[INFO] Đang tải VnCoreNLP (tách từ + gán nhãn từ loại + phân tích cú pháp)...")
        self.annotator = load_segmenter(annotators=("wseg", "pos", "ner", "parse"))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.device = get_device()

        print("[INFO] Đang xây dựng mạng PhoBERT + GCN...")
        self.trained = os.path.exists(os.path.join(model_dir, "model.safetensors"))
        if self.trained:
            self.model = PhoBertGCN.from_trained(model_dir, self.device)
            print(f"[INFO] Đã nạp trọng số đã huấn luyện: {model_dir}")
        else:
            self.model = PhoBertGCN(MODEL_NAME, num_labels=len(id2label)).to(self.device).eval()
            print(f"[CẢNH BÁO] Không tìm thấy trọng số trong {model_dir}. Lớp GCN đang ở trạng thái khởi tạo ngẫu nhiên, "
                  "nhãn dự đoán và màu sắc sẽ KHÔNG có ý nghĩa. Hãy chạy train_gcn.py trước.")
        self.model.requires_grad_(False)
        # PhoBERT gốc (chưa fine-tune) để so sánh ma trận attention trước / sau huấn luyện
        self.pretrained = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(self.device).eval()
        self.top_k = top_k
        self.min_attn = min_attn

    # ---------- Dự đoán ----------
    @torch.no_grad()
    def predict(self, texts):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(self.device)
        logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def encode(self, words):
        """Mã hóa từng từ thành subword; trả về (input_ids, vị trí [đầu, cuối) của mỗi từ)."""
        spans, ids = [], [self.tokenizer.cls_token_id]
        for w in words:
            pieces = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(w)) or [self.tokenizer.unk_token_id]
            if len(ids) + len(pieces) > MAX_LENGTH - 1:
                break
            spans.append((len(ids), len(ids) + len(pieces)))
            ids.extend(pieces)
        ids.append(self.tokenizer.sep_token_id)
        return torch.tensor([ids], device=self.device), spans

    @torch.no_grad()
    def word_attention(self, words, pretrained=False):
        """Ma trận kề A (attention layer cuối, trung bình các head), gộp subword lên mức từ (hàng: trung bình, cột: cộng)."""
        input_ids, spans = self.encode(words)
        mask = torch.ones_like(input_ids)
        if pretrained:
            A = self.pretrained(input_ids=input_ids, attention_mask=mask, output_attentions=True).attentions[-1].mean(dim=1)
        else:
            A = self.model(input_ids=input_ids, attention_mask=mask, return_graph=True).attentions[0]
        A = A[0].cpu().numpy()

        W = np.zeros((len(words), len(words)))
        for i, (si, ei) in enumerate(spans):
            for j, (sj, ej) in enumerate(spans):
                W[i, j] = A[si:ei, sj:ej].sum(axis=1).mean()
        return W

    def integrated_gradients(self, words, steps=32):
        """Đóng góp của từng từ vào P(tích cực) - P(tiêu cực) bằng Integrated Gradients.
        Embedding đi dần từ <pad> (baseline) tới embedding thật; tổng đóng góp = f(câu) - f(baseline),
        nên không bị bão hòa như cách xóa từng từ khi câu có nhiều từ cùng cảm xúc."""
        input_ids, spans = self.encode(words)
        emb_layer = self.model.phobert.embeddings.word_embeddings
        baseline = emb_layer.weight[self.tokenizer.pad_token_id]
        alphas = torch.linspace(1.0 / steps, 1.0, steps, device=self.device).view(-1, 1, 1)
        special = ((input_ids == self.tokenizer.cls_token_id) | (input_ids == self.tokenizer.sep_token_id)).unsqueeze(-1)
        store = {}

        def hook(module, inputs, out):
            emb = out.detach()
            scaled = (baseline + alphas * (emb - baseline)).requires_grad_(True)
            store["scaled"], store["emb"] = scaled, emb
            return torch.where(special, emb, scaled)  # giữ nguyên <s>, </s>

        handle = emb_layer.register_forward_hook(hook)
        try:
            ids = input_ids.expand(steps, -1)
            probs = torch.softmax(self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits, dim=-1)
            (probs[:, 2] - probs[:, 0]).sum().backward()
        finally:
            handle.remove()
        grads = store["scaled"].grad.mean(dim=0)                        # (seq_len, hidden)
        attr = ((store["emb"][0] - baseline) * grads).sum(dim=-1).cpu().numpy()
        return [float(attr[s:e].sum()) for s, e in spans] + [0.0] * (len(words) - len(spans))

    # ---------- Xây đồ thị cho 1 câu ----------
    def annotate(self, sentence):
        """Trả về list từ: dict(index, wordForm, posTag, head, depLabel), index/head đánh số liên tục từ 1."""
        result = self.annotator.annotate_text(sentence)
        tokens, offset = [], 0
        for sent_id in sorted(result):
            sent = result[sent_id]
            for t in sent:
                tokens.append(dict(index=t["index"] + offset, wordForm=t["wordForm"], posTag=t["posTag"],
                                   head=t["head"] + offset if t["head"] > 0 else 0, depLabel=t["depLabel"]))
            offset += len(sent)
        return tokens

    def split_clauses(self, tokens):
        """Tách câu thành các mệnh đề tại dấu phẩy / từ nối; trả về (danh sách mệnh đề, danh sách từ nối)."""
        clauses, connectives, cur = [], [], []
        for t in tokens:
            w = t["wordForm"].lower()
            if (w in CLAUSE_CONNECTIVES and cur) or w in CLAUSE_PUNCT:
                if cur:
                    clauses.append(cur)
                    connectives.append(display(t["wordForm"]) if w in CLAUSE_CONNECTIVES else "")
                    cur = []
                elif w in CLAUSE_CONNECTIVES and connectives:
                    connectives[-1] = display(t["wordForm"])
                continue
            if t["posTag"] != "CH":
                cur.append(t)
        if cur:
            clauses.append(cur)
        return clauses, connectives[:max(len(clauses) - 1, 0)]

    def build(self, sentence, sid):
        sentence = unicodedata.normalize("NFC", sentence.strip())
        tokens = self.annotate(sentence)
        words = [t["wordForm"] for t in tokens]
        content = [t for t in tokens if t["posTag"] != "CH"]
        clauses, connectives = self.split_clauses(tokens)

        # Dự đoán cả câu và từng mệnh đề trong 1 lần chạy; đóng góp của từ tính bằng Integrated Gradients
        probs = self.predict([" ".join(words)] + [" ".join(t["wordForm"] for t in c) for c in clauses])
        sent_p, clause_p = probs[0], probs[1:]
        attributions = self.integrated_gradients(words)
        importance = {t["index"]: a for t, a in zip(tokens, attributions)}

        kg = KnowledgeGraph()
        node_id = {}

        # Tầng 0: câu
        label = int(sent_p.argmax())
        root_id = f"s{sid}"
        kg.nodes.append(Node(id=root_id, level=0, shape="box", color=SENTIMENT_COLORS[label],
                             label=f"Câu {sid}: {wrap(sentence, 30)}\n{id2label[label]} {sent_p[label] * 100:.0f}%"))

        # Tầng 2: từ (danh từ là thực thể dùng chung giữa các câu); từ nối mệnh đề đã thành nhãn cạnh
        for t in (t for c in clauses for t in c):
            w = t["wordForm"]
            if t["posTag"] in ("N", "Np", "Nc", "Ny"):
                nid = "e_" + w.lower()
                kg.nodes.append(Node(id=nid, level=2, shape="box", color=ENTITY_COLOR, label=display(w)))
            else:
                nid = f"s{sid}w{t['index']}"
                s = importance[t["index"]]
                kg.nodes.append(Node(id=nid, level=2, color=score_color(s), label=f"{display(w)}\n{s:+.2f}"))
            node_id[t["index"]] = nid

        # Tầng 1: mệnh đề + cây cú pháp bên trong mỗi mệnh đề
        multi = len(clauses) > 1
        for ci, clause in enumerate(clauses):
            p = clause_p[ci]
            parent = root_id
            if multi:
                parent = f"s{sid}c{ci}"
                text = " ".join(display(t["wordForm"]) for t in clause)
                kg.nodes.append(Node(id=parent, level=1, color=SENTIMENT_COLORS[int(p.argmax())],
                                     label=f"Mệnh đề: {wrap(text, 24)}\n{id2label[int(p.argmax())]} {p.max() * 100:.0f}%"))
                kg.edges.append(Edge(source=root_id, target=parent, label=f"mệnh đề {ci + 1}", color=SYNTAX_COLOR))
                if ci > 0:
                    kg.edges.append(Edge(source=f"s{sid}c{ci - 1}", target=parent, color="#8e44ad",
                                         label=connectives[ci - 1] or "nối tiếp", style="bold"))
            members = {t["index"] for t in clause}
            for t in clause:
                head_in = t["head"] in members
                src = node_id[t["head"]] if head_in else parent
                rel = DEP_LABELS.get(t["depLabel"], t["depLabel"]) if head_in else "gốc"
                if src != node_id[t["index"]]:
                    kg.edges.append(Edge(source=src, target=node_id[t["index"]], label=rel, color=SYNTAX_COLOR))

        # Cạnh attention: top-k liên kết mạnh nhất của mỗi từ (chuẩn hóa lại sau khi bỏ dấu câu)
        W = self.word_attention(words)
        keep = [i for i, t in enumerate(tokens) if t["posTag"] != "CH"]
        sub = W[np.ix_(keep, keep)]
        np.fill_diagonal(sub, 0)
        sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-9, None)
        syntax = {frozenset((e.source, e.target)): k for k, e in enumerate(kg.edges)}
        added, matched = set(), {}
        for a, i in enumerate(keep):
            for b in np.argsort(-sub[a])[:self.top_k]:
                if sub[a, b] < self.min_attn:
                    continue
                src, dst = node_id.get(tokens[i]["index"]), node_id.get(tokens[keep[b]]["index"])
                if not src or not dst or src == dst or (src, dst) in added:
                    continue
                added.add((src, dst))
                k = syntax.get(frozenset((src, dst)))
                if k is not None:
                    matched.setdefault(k, []).append(sub[a, b])
                else:
                    kg.edges.append(Edge(source=src, target=dst, label=f"{sub[a, b]:.2f}",
                                         color=ATTN_COLOR, style="dashed"))

        # Attention trùng với quan hệ cú pháp: ghi trọng số (cả hai chiều nếu có) lên chính cạnh cú pháp đó
        for k, ws in matched.items():
            e = kg.edges[k]
            attn = " / ".join(f"{w:.2f}" for w in ws)
            kg.edges[k] = e.model_copy(update={"label": f"{e.label}\nattn {attn}", "color": ATTN_COLOR})

        heat = ([display(w) for w in words], self.word_attention(words, pretrained=True), W, sentence, id2label[label])
        return kg, heat


def draw_heatmap(labels, before, after, sentence, label, path):
    """Hai ma trận kề mức từ cạnh nhau: PhoBERT gốc và sau fine-tune.
    Cột cuối là phần attention dồn vào <s>, </s> (không phải từ nào trong câu)."""
    cols = labels + ["<s>, </s>"]
    mats = [np.hstack([m, 1 - m.sum(axis=1, keepdims=True)]) for m in (before, after)]
    vmax = max(m.max() for m in mats)
    n = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(2 + 1.2 * (n + 1), 1.5 + 0.55 * n))
    titles = ["PhoBERT gốc (chưa fine-tune)", "Sau fine-tune (ma trận kề GCN thực sự dùng)"]
    for ax, m, title in zip(axes, mats, titles):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        ax.axvline(n - 0.5, color="#e67e22", lw=1.5)
        ax.set_xticks(range(n + 1), cols, rotation=45, ha="right")
        ax.set_yticks(range(n), labels)
        for i in range(n):
            for j in range(n + 1):
                if m[i, j] >= 0.05:
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if m[i, j] > vmax * 0.6 else "black")
        ax.set_xlabel("Từ được chú ý (đích)")
        ax.set_title(f"{title}\nattention vào <s>, </s>: {m[:, -1].mean() * 100:.0f}%", fontsize=10)
    axes[0].set_ylabel("Từ nguồn")
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(f"Ma trận kề mức từ (attention layer cuối, trung bình 12 head)\n{sentence}  →  {label}", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 3. CẬP NHẬT LẶP QUA NHIỀU CÂU
# ==========================================
def generate_graph(sentences: List[str], builder: GraphBuilder, out_dir: str) -> KnowledgeGraph:
    cur_state = KnowledgeGraph()
    for i, sentence in enumerate(sentences, start=1):
        print(f"[INFO] ({i}/{len(sentences)}) {sentence}")
        new_updates, heat = builder.build(sentence, sid=i)
        draw_heatmap(*heat, os.path.join(out_dir, f"attention_heatmap_{i}.png"))
        cur_state = cur_state.update(new_updates)
        cur_state.title = f"Đồ thị tri thức PhoBERT + GCN (sau {i}/{len(sentences)} câu)"
        if not builder.trained:
            cur_state.title += " [GCN CHƯA HUẤN LUYỆN]"
        cur_state.draw(os.path.join(out_dir, f"iteration_{i}"))
    return cur_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trực quan hóa đồ thị tri thức PhoBERT + GCN")
    parser.add_argument("sentences", nargs="*", help="Các câu cần phân tích (mặc định: câu mẫu)")
    parser.add_argument("--model-dir", default=str(BEST_MODEL_DIR), help="Thư mục chứa model.safetensors đã huấn luyện")
    parser.add_argument("--top-k", type=int, default=1, help="Số cạnh attention giữ lại cho mỗi từ")
    parser.add_argument("--min-attn", type=float, default=0.15, help="Ngưỡng trọng số attention tối thiểu")
    parser.add_argument("--out", default=str(RESULTS_DIR / "graphs"), help="Thư mục lưu ảnh đầu ra")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    builder = GraphBuilder(args.model_dir, top_k=args.top_k, min_attn=args.min_attn)
    graph = generate_graph(args.sentences or DEFAULT_SENTENCES, builder, args.out)
    path = graph.draw(os.path.join(args.out, "final"))
    print(f"--- Đã vẽ xong đồ thị: {path} (cùng các file iteration_*.png, attention_heatmap_*.png trong {args.out}/) ---")
