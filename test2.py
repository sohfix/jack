from torch.distributed import tensor
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict
import torch
from models.Blender import LoadingBar as loadb

loading = loadb()
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f"Using device: {device}")

# Load dataset
data = "vicgalle/alpaca-gpt4"  # Replace with your dataset
dataset = load_dataset(data)


loading.start(' ')
# Wait for user input to proceed (useful for checking dataset keys)
input("Press Enter to continue...")
loading.stop()

# Load the tokenizer and model
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained(
    "facebook/blenderbot-400M-distill"
).to(device)


# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


# Check if 'validation' split exists, if not create one
if "validation" not in dataset.keys():
    train_test_split = dataset["train"].train_test_split(test_size=0.1)
    dataset = DatasetDict(
        {"train": train_test_split["train"], "validation": train_test_split["test"]}
    )

print('Tokenizing dataset')
loading.start('tokens')
tokenized_dataset = dataset.map(preprocess_function, batched=True)
loading.stop()

print('Training arguments')
# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
)

print('__init__.trainer()')
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train the model
trainer.train()
