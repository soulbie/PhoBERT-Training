"""
Huấn luyện mô hình PhoBERT + GCN phân loại cảm xúc (Tiêu cực / Trung tính / Tích cực).

Cách dùng:
  python train_gcn.py
  python train_gcn.py --epochs 5 --lr 2e-5 --batch-size 32

Đầu ra:
  phobert_gcn_model/best_model/      trọng số tốt nhất (theo F1 trên tập validation) + tokenizer
  results/metrics.json               chỉ số đánh giá của best_model + lịch sử huấn luyện
  results/loss_curve.png             biểu đồ Training / Validation Loss
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Không cần màn hình (chạy được trên server / Colab)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from model import (DATA_DIR, MAX_LENGTH, MODEL_DIR, MODEL_NAME, RESULTS_DIR, PhoBertGCN, label2id,
                   load_segmenter, segment_text)

sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện PhoBERT + GCN phân loại cảm xúc tiếng Việt")
    parser.add_argument("--train-file", default=str(DATA_DIR / "synthetic_train.csv"))
    parser.add_argument("--val-file", default=str(DATA_DIR / "synthetic_val.csv"))
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR, help="Nơi lưu checkpoint và best_model/")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Nơi lưu metrics.json, loss_curve.png")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5, help="LR cao hơn mức thường dùng vì lớp GCN khởi tạo ngẫu nhiên")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Giữ lại các checkpoint từng epoch (mặc định xóa sau khi train, mỗi cái ~1.6GB)")
    return parser.parse_args()


def load_data(path, segmenter):
    df = pd.read_csv(path)
    unknown = set(df["sentiment"]) - set(label2id)
    if unknown:
        raise ValueError(f"{path}: nhãn không hợp lệ {unknown}, chỉ chấp nhận {list(label2id)}")
    df["label"] = df["sentiment"].map(label2id)
    df["segmented_text"] = df["sentence"].apply(lambda s: segment_text(segmenter, s))
    return Dataset.from_pandas(df[["segmented_text", "label"]], preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": precision, "recall": recall, "f1": f1}


def plot_loss(log_history, path):
    train = [(log["step"], log["loss"]) for log in log_history if "loss" in log]
    val = [(log["step"], log["eval_loss"]) for log in log_history if "eval_loss" in log]

    plt.figure(figsize=(10, 6))
    if train:
        plt.plot(*zip(*train), label="Training Loss", color="blue", linewidth=2)
    if val:
        plt.plot(*zip(*val), label="Validation Loss", color="red", linestyle="--", marker="o", linewidth=2)
    plt.title("Biểu đồ Loss - Mô hình PhoBERT + GCN")
    plt.xlabel("Số bước (Steps)")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("--- [1/4] Khởi tạo VnCoreNLP và tách từ dữ liệu ---")
    segmenter = load_segmenter()
    train_ds = load_data(args.train_file, segmenter)
    val_ds = load_data(args.val_file, segmenter)
    print(f"Train: {len(train_ds)} câu | Validation: {len(val_ds)} câu")

    print("--- [2/4] Tokenize ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["segmented_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    print("--- [3/4] Huấn luyện PhoBERT + GCN ---")
    model = PhoBertGCN(MODEL_NAME, num_labels=len(label2id))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,  # Bắt buộc: Trainer không được tự xóa cột đầu vào của custom model
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("--- [4/4] Đánh giá best model và lưu kết quả ---")
    # Đánh giá lại sau khi đã nạp best model, để số liệu báo cáo khớp đúng với trọng số được lưu
    final_metrics = trainer.evaluate()
    best_model_dir = args.output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"model_name": MODEL_NAME, "max_length": MAX_LENGTH},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "best_model_metrics": final_metrics,
        "log_history": trainer.state.log_history,
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_loss(trainer.state.log_history, args.results_dir / "loss_curve.png")

    if not args.keep_checkpoints:
        for ckpt in args.output_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt)

    print(f"Best model ({trainer.state.best_model_checkpoint}): "
          f"accuracy={final_metrics['eval_accuracy']:.4f}, f1={final_metrics['eval_f1']:.4f}")
    print(f"Đã lưu mô hình vào {best_model_dir}, kết quả vào {args.results_dir}")


if __name__ == "__main__":
    main()
