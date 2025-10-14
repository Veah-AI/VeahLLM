"""Training script for VEAH LLM"""

import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def train_model(model_name="veah-7b", dataset_path="data/solana_docs"):
    print(f"Training {model_name} on {dataset_path}")

    # Load dataset
    dataset = load_dataset("json", data_files=f"{dataset_path}/*.jsonl")

    # Training args
    args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=1000,
        learning_rate=2e-5,
        fp16=True,
        save_steps=5000,
    )

    # Train
    trainer = Trainer(args=args, train_dataset=dataset["train"])
    trainer.train()

    return trainer.model

if __name__ == "__main__":
    train_model()