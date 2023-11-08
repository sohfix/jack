from printy import inputy
from Blender import LoadingBar, BlenderbotPipeline, MODELS

# facebook
# todo run
loading_bar = LoadingBar()
model = MODELS["B"][1]

loading_bar.start(model)

try:
    pipe = BlenderbotPipeline(model)
finally:
    loading_bar.stop()

response = pipe(inputy(f"[c]Welcome to the {model}, ask away: @"))

# Print the generated response
print(response)
