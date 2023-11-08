from transformers import pipeline

# Initialize the pipeline for text generation with the gpt2-medium model
pipe = pipeline("text-generation", model="gpt2-medium")

text = ""

# Provide a prompt for text generation
prompt = f"{text}"

generated_text = pipe(prompt, max_length=100, num_return_sequences=3)

# Print the generated text
for output in generated_text:
    print(output["generated_text"])
