from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

print(f"Using device: {device}")

# Load dataset
data = "vicgalle/alpaca-gpt4"
dataset = load_dataset(data)
print(dataset.keys())
input()

# Load the tokenizer and model
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained(
    "facebook/blenderbot-400M-distill"
).to(device)


# Tokenize the dataset (adjust based on your dataset structure)
def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    evaluation_strategy="steps",  # Add this line
    save_strategy="steps",  # And this line
    eval_steps=500,  # And set this to determine how often to evaluate
    save_steps=500,  # And how often to save
)

# Init Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train the model
trainer.train()
