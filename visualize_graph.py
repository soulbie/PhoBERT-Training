import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import py_vncorenlp
import networkx as nx
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

# 1. Khởi tạo công cụ
print("Đang khởi tạo VnCoreNLP và PhoBERT...")
vncorenlp_dir = os.path.abspath("./vncorenlp")
if not os.path.exists(vncorenlp_dir):
    print("Chưa có VnCoreNLP, đang tiến hành tải xuống...")
    os.makedirs(vncorenlp_dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=vncorenlp_dir)

original_cwd = os.getcwd()
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
os.chdir(original_cwd)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert = AutoModel.from_pretrained("vinai/phobert-base", attn_implementation="eager")
phobert.eval()

# 2. Xử lý câu văn
sentence = "Giảng viên này dạy rất tuyệt nhưng hay đến muộn."
segmented_sentence = " ".join(rdrsegmenter.word_segment(sentence))
print(f"Câu sau khi tách từ: {segmented_sentence}")

# 3. Tokenize và chạy qua PhoBERT để lấy Attention
inputs = tokenizer(segmented_sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = phobert(**inputs, output_attentions=True)
    
# 4. Trích xuất Ma trận Attention từ layer cuối cùng
last_layer_attn = outputs.attentions[-1] # (1, 12_heads, seq_len, seq_len)
attn_matrix = last_layer_attn.mean(dim=1).squeeze(0).cpu().numpy() # (seq_len, seq_len)

# 5. Xây dựng đồ thị bằng NetworkX
G = nx.DiGraph()

# Thêm Node (là các từ vựng)
num_tokens = len(tokens)
for i in range(num_tokens):
    # Lọc bớt các ký hiệu đặc biệt để đồ thị nhìn sạch hơn
    if tokens[i] not in ['<s>', '</s>']:
        G.add_node(i, label=tokens[i].replace("_", " "))

# Thêm Edge (Cạnh) dựa trên độ lớn của Attention
# Chỉ lấy những kết nối có trọng số đủ lớn (ví dụ > 0.05) để đồ thị không bị rối
threshold = 0.05
for i in range(1, num_tokens - 1): # Bỏ qua <s> và </s>
    for j in range(1, num_tokens - 1):
        weight = attn_matrix[i, j]
        if weight > threshold and i != j:
            G.add_edge(i, j, weight=weight)

# 6. Vẽ đồ thị
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.8, seed=42) # Thuật toán tự dãn khoảng cách các nút

# Vẽ nút
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.9, edgecolors='gray')

# Vẽ nhãn nút
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold')

# Vẽ cạnh với độ đậm nhạt tương ứng trọng số
edges = G.edges()
weights = [G[u][v]['weight'] * 10 for u, v in edges] # Nhân 10 để nét vẽ rõ hơn
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray', arrows=True, arrowsize=15)

plt.title("Trực quan hóa cấu trúc Đồ thị Attention của PhoBERT\n(Độ dày của mũi tên thể hiện mức độ liên kết ngữ nghĩa)", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('attention_graph_visualization.png', dpi=300)
print("--- Đã vẽ thành công đồ thị vào file: attention_graph_visualization.png ---")
