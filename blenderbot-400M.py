from printy import inputy
from Blender import LoadingBar, BlenderbotPipeline, MODELS

loading_bar = LoadingBar()
model = MODELS['M'][400]

loading_bar.start(model)

try:
    pipe = BlenderbotPipeline(model)
finally:
    loading_bar.stop()

response = pipe(inputy(f"[c]Welcome to the {model}, ask away: @"))

# Print the generated response
print(response)
# h
