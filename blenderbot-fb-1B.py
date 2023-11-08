# Initialize the loading bar
loading_bar = LoadingBar()
model = "facebook/blenderbot-1B-distill"
loading_bar.start(model)

try:
    pipe = BlenderbotPipeline(model)
finally:
    loading_bar.stop()

response = pipe(inputy(f"[c]Welcome to the {model}, ask away: @"))

# Print the generated response
print(response)
