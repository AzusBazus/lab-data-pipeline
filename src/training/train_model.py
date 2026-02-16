from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk
import torch
from config import LABELS, BASE_MODEL_PATH, DATASET_PATH, MODEL_PATH

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
        output_dir=MODEL_PATH,
        max_steps=200,                # Short run for testing
        per_device_train_batch_size=2, # Keep small for CPU/MPS
        learning_rate=5e-5,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=50,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_pin_memory=False
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