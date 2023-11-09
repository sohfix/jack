import time
from printy import printy
from Blender import BlenderbotPipeline, MODELS, CUDA
from models.mutil import ModelPrinter, ExecutionTimer, clear

x = 1
timer, printer = ExecutionTimer(), ModelPrinter()
timer.start()
printy(CUDA, "y")
model = MODELS["M"][400]


try:
    pipe = BlenderbotPipeline(model)
finally:
    _ = [print("*", end="") for i in range(45)]
    time.sleep(1)

# clear()

for i in range(x):
    response = pipe(
        "sailing in a storm",
        max_length=200,
        min_length=100,
        length_penalty=2.0,
        num_beams=6,
    )
    # clear()
    printy(f"\n{i + 1}", "y")
    printer.print(response)
timer.stop()
