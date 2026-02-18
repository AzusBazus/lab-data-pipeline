from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk
import torch
from src.config import LABELS, BASE_MODEL_PATH, DATASET_PATH, MODEL_PATH

id2label = {k: v for k, v in enumerate(LABELS)}
label2id = {v: k for k, v in enumerate(LABELS)}

def main():
    print("⏳ Loading Dataset...")
    dataset = load_from_disk(DATASET_PATH)
    
    # Split: 80% Train, 20% Test (even with 20 items, we need to verify overfitting)
    dataset = dataset.train_test_split(test_size=0.2)
    
    print(f"🏋️‍♀️ Training on {len(dataset['train'])} examples...")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL_PATH,
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=f"{MODEL_PATH}/final",
        
        # CHANGE 1: Use Epochs, not Steps
        max_steps=-1,               # Disable step limit
        num_train_epochs=20,        # Train for 20 full cycles (Standard for LayoutLM)
        
        # CHANGE 2: Maintenance
        per_device_train_batch_size=4, # Keep low for 249 samples
        save_strategy="epoch",      # Save model at end of every epoch
        eval_strategy="epoch",# Check performance every epoch
        logging_strategy="epoch",   # Less log spam
        load_best_model_at_end=True,# Automatically load the best epoch at the end
        metric_for_best_model="eval_loss", # optimizing for F1 score (accuracy)
        greater_is_better=False,  # Lower loss is better
        save_total_limit=3,         # Only keep the top 3 checkpoints (saves disk space)
        warmup_ratio=0.1,           # Warmup helps with new data
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    print("🚀 Starting Training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(MODEL_PATH + "/final")
    print("✅ Training Complete! Model saved.")

if __name__ == "__main__":
    main()