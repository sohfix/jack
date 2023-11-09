from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data = "vicgalle/alpaca-gpt4"  # Replace with your dataset
dataset = load_dataset(data)

# Load the tokenizer and model
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained(
    "facebook/blenderbot-400M-distill"
).to(device)


# Tokenize the dataset (adjust based on your dataset structure)
def preprocess_function(examples):
    return tokenizer(
        examples["input_text"], truncation=True, padding="max_length", max_length=512
    )


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Output directory for model checkpoints
    num_train_epochs=1,  # Total number of training epochs
    per_device_train_batch_size=8,  # Batch size per device during training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir="./logs",  # Directory for storing logs
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train the model
trainer.train()
