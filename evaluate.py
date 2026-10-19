"""
Đánh giá mô hình đã huấn luyện trên một tập dữ liệu có nhãn (mặc định: tập validation).

Cách dùng:
  python evaluate.py
  python evaluate.py --data data/synthetic_val.csv --model-dir phobert_gcn_model/best_model

Đầu ra:
  In ra classification report theo từng nhãn và theo từng chủ đề (topic)
  results/evaluation.json          chỉ số tổng hợp + theo nhãn + theo chủ đề
  results/confusion_matrix.png     ma trận nhầm lẫn
"""
import argparse
import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from transformers import AutoTokenizer

from model import (BEST_MODEL_DIR, DATA_DIR, MAX_LENGTH, RESULTS_DIR, PhoBertGCN, id2label, label2id,
                   load_segmenter, segment_text)

sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger("transformers").setLevel(logging.ERROR)


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size):
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_LENGTH).to(model.classifier.weight.device)
        preds.extend(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.argmax(-1).tolist())
    return preds


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình PhoBERT + GCN")
    parser.add_argument("--data", default=str(DATA_DIR / "synthetic_val.csv"))
    parser.add_argument("--model-dir", default=str(BEST_MODEL_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    segmenter = load_segmenter()
    texts = [segment_text(segmenter, s) for s in df["sentence"]]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = PhoBertGCN.from_trained(args.model_dir)

    y_true = df["sentiment"].map(label2id).tolist()
    y_pred = predict(model, tokenizer, texts, args.batch_size)
    df["pred"] = y_pred
    names = [id2label[i] for i in range(len(id2label))]

    print(f"\n=== Kết quả trên {args.data} ({len(df)} câu) ===")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    report = {
        "data": args.data,
        "num_samples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "per_class": classification_report(y_true, y_pred, target_names=names, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if "topic" in df:
        print("=== Accuracy theo chủ đề ===")
        report["per_topic_accuracy"] = {}
        for topic, g in df.groupby("topic"):
            acc = accuracy_score(g["sentiment"].map(label2id), g["pred"])
            report["per_topic_accuracy"][topic] = acc
            print(f"  {topic:<12} {acc:.4f}  ({len(g)} câu)")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Ma trận nhầm lẫn - PhoBERT + GCN")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    print(f"\nĐã lưu {RESULTS_DIR / 'evaluation.json'} và {RESULTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
