from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
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
        max_steps=-1,
        num_train_epochs=20,
        
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        fp16=False,         
        dataloader_num_workers=0,

        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        
        warmup_ratio=0.1,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],

        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("🚀 Starting Training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(MODEL_PATH + "/final")
    print("✅ Training Complete! Model saved.")

if __name__ == "__main__":
    main()