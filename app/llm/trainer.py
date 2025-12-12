"""
Fine-tuning script for Kempian LLM
Trains the model on HR/recruiting domain data
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk, Dataset
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_MODEL = os.getenv("LLM_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
OUTPUT_DIR = os.getenv("LLM_OUTPUT_DIR", "./models/kempian-llm-v1.0")
TRAIN_DATA = os.getenv("LLM_TRAIN_DATA", "./data/train_dataset")
VAL_DATA = os.getenv("LLM_VAL_DATA", "./data/val_dataset")
DEVICE = os.getenv("LLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

def load_training_data(data_path: str) -> Dataset:
    """Load training data from disk or JSON file"""
    if os.path.isdir(data_path):
        # Load from saved dataset
        return load_from_disk(data_path)
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        # Load from JSON and convert
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted = []
        for item in data:
            if isinstance(item, dict):
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                
                # Combine into training text
                full_text = f"{instruction}\n\n{input_text}\n\n{output_text}"
                formatted.append({"text": full_text})
        
        return Dataset.from_list(formatted)
    else:
        raise FileNotFoundError(f"Training data not found: {data_path}")

def train():
    """Main training function"""
    logger.info("=" * 50)
    logger.info("Starting Kempian LLM Fine-Tuning")
    logger.info("=" * 50)
    
    # Load datasets
    logger.info(f"Loading training data from: {TRAIN_DATA}")
    train_dataset = load_training_data(TRAIN_DATA)
    
    if os.path.exists(VAL_DATA):
        logger.info(f"Loading validation data from: {VAL_DATA}")
        val_dataset = load_training_data(VAL_DATA)
    else:
        logger.info("No validation data found, splitting training data")
        train_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
        val_dataset = train_dataset["test"]
        train_dataset = train_dataset["train"]
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model from: {BASE_MODEL}")
    logger.info(f"Device: {DEVICE}")
    
    quantization_config = None
    if DEVICE == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("Using 4-bit quantization")
    
    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True
        )
        if DEVICE != "cuda":
            model = model.to(DEVICE)
    
    # Setup LoRA for parameter-efficient fine-tuning
    logger.info("Setting up LoRA for efficient fine-tuning")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt"
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names if hasattr(train_dataset, 'column_names') else ["text"]
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names if hasattr(val_dataset, 'column_names') else ["text"]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=(DEVICE == "cuda"),
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=100,
        report_to="tensorboard",
        load_best_model_at_end=True,
        save_total_limit=3
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # Train
    logger.info("=" * 50)
    logger.info("Training started...")
    logger.info("=" * 50)
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(OUTPUT_DIR, "final")
    logger.info(f"Saving final model to: {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Log training metrics
    metrics = trainer.state.log_history
    if metrics:
        logger.info("Training completed!")
        logger.info(f"Final training loss: {metrics[-1].get('train_loss', 'N/A')}")
        logger.info(f"Final validation loss: {metrics[-1].get('eval_loss', 'N/A')}")
    
    logger.info("=" * 50)
    logger.info(f"Training complete! Model saved to: {final_output_dir}")
    logger.info("=" * 50)
    logger.info("Next steps:")
    logger.info(f"1. Update LLM_MODEL_PATH in .env to: {final_output_dir}")
    logger.info("2. Restart backend server")
    logger.info("3. Test the fine-tuned model")
    
    return trainer

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

