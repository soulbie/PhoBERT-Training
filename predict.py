"""
Dự đoán cảm xúc cho câu tiếng Việt bằng mô hình PhoBERT + GCN đã huấn luyện.

Cách dùng:
  python predict.py                                   # chế độ tương tác, gõ 'exit' để thoát
  python predict.py "Giảng viên dạy rất dễ hiểu."     # dự đoán trực tiếp một hoặc nhiều câu
"""
import argparse
import logging
import sys

import torch
from transformers import AutoTokenizer

from model import BEST_MODEL_DIR, MAX_LENGTH, PhoBertGCN, id2label, load_segmenter, segment_text

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger("transformers").setLevel(logging.ERROR)


class SentimentPredictor:
    def __init__(self, model_dir=BEST_MODEL_DIR):
        print("[INFO] Đang tải VnCoreNLP...")
        self.segmenter = load_segmenter()
        print("[INFO] Đang tải tokenizer và mô hình PhoBERT + GCN...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = PhoBertGCN.from_trained(model_dir)
        print(f"[INFO] Sẵn sàng (thiết bị: {self.model.classifier.weight.device}).")

    @torch.no_grad()
    def __call__(self, text):
        """Trả về (nhãn, độ tin cậy, xác suất của từng nhãn)."""
        enc = self.tokenizer(segment_text(self.segmenter, text), return_tensors="pt", truncation=True,
                             max_length=MAX_LENGTH).to(self.model.classifier.weight.device)
        probs = torch.softmax(self.model(**enc).logits, dim=-1)[0].cpu()
        label_id = int(probs.argmax())
        return id2label[label_id], float(probs[label_id]), {id2label[i]: float(p) for i, p in enumerate(probs)}


def show(text, predictor):
    label, conf, probs = predictor(text)
    detail = "  ".join(f"{name}: {p * 100:5.1f}%" for name, p in probs.items())
    print(f"  => {label:<11} ({conf * 100:.2f}%) | {detail}\n")


def main():
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc câu tiếng Việt bằng PhoBERT + GCN")
    parser.add_argument("sentences", nargs="*", help="Các câu cần dự đoán (bỏ trống để vào chế độ tương tác)")
    parser.add_argument("--model-dir", default=str(BEST_MODEL_DIR))
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model_dir)

    if args.sentences:
        for s in args.sentences:
            print(f"> {s}")
            show(s, predictor)
        return

    print("\nGõ câu cần phân tích, 'exit' hoặc 'quit' để thoát.\n")
    while True:
        try:
            text = input("> Text: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("thoát", "exit", "quit", "q"):
            break
        if text:
            show(text, predictor)
    print("\n[INFO] Đã thoát chương trình.")


if __name__ == "__main__":
    main()
