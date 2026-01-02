"""
Training script for SageMaker
This runs inside SageMaker training container
"""

import argparse
import os
import json
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
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker passes hyperparameters as arguments
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    
    # SageMaker passes data paths via environment variables
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_args()

def prepare_dataset(data_path):
    """Load and prepare training dataset"""
    logger.info(f"Loading data from: {data_path}")
    
    # Find training data file
    if os.path.isfile(data_path):
        data_file = data_path
    elif os.path.isdir(data_path):
        # Look for JSON file in directory
        json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
        if json_files:
            data_file = os.path.join(data_path, json_files[0])
        else:
            raise FileNotFoundError(f"No JSON file found in {data_path}")
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Load JSON data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {data_file}")
    
    # Format for training
    formatted = []
    for item in data:
        if isinstance(item, dict):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # Combine into training text
            full_text = f"{instruction}\n\n{input_text}\n\n{output_text}"
            formatted.append({"text": full_text})
    
    # Create dataset
    dataset = Dataset.from_list(formatted)
    
    # Split train/validation
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    logger.info(f"Training examples: {len(dataset['train'])}")
    logger.info(f"Validation examples: {len(dataset['test'])}")
    
    return dataset['train'], dataset['test']

def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting SageMaker Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.train_batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info()
    
    # Load datasets
    train_dataset, val_dataset = prepare_dataset(args.train)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    logger.info("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
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
            padding="max_length"
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=100,
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="tensorboard"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # Train
    logger.info("=" * 60)
    logger.info("Training started...")
    logger.info("=" * 60)
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Log final metrics
    metrics = trainer.state.log_history
    if metrics:
        final_metrics = metrics[-1]
        logger.info("=" * 60)
        logger.info("Training Metrics:")
        logger.info(f"  Final Training Loss: {final_metrics.get('train_loss', 'N/A')}")
        logger.info(f"  Final Validation Loss: {final_metrics.get('eval_loss', 'N/A')}")
        logger.info("=" * 60)
    
    logger.info("=" * 60)
    logger.info("✅ Training complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

