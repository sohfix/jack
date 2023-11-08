from printy import inputy
from Blender import LoadingBar, BlenderbotPipeline, MODELS

# todo run

loading_bar = LoadingBar()
model = MODELS["M"][400]

loading_bar.start(model)

try:
    pipe = BlenderbotPipeline(model)
finally:
    loading_bar.stop()

response = pipe(
    inputy(f"[c]Welcome to the {model}, ask away: @"),
    max_length=100,
    min_length=50,
    length_penalty=2.0,
    num_beams=6,
)

# Print the generated response
print(response)
# h
