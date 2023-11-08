from transformers import pipeline

# Initialize the pipeline for text generation with the gpt2-medium model
pipe = pipeline("text-generation", model="gpt2-medium")

# Provide a prompt for text generation
prompt = "The adventures of Jack and John were always filled with wonder."

# Generate text using the pipeline
generated_text = pipe(prompt, max_length=200, num_return_sequences=5)

# Print the generated text
for output in generated_text:
    print(output['generated_text'])
    print(output)
