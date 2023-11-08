import threading
import itertools
import time
import sys
from printy import inputy, printy


from transform import BlenderbotPipeline


class LoadingBar:
    def __init__(self, delay=0.1):
        self.delay = delay  # In seconds
        self.spinner_signs = itertools.cycle(["-", "\\", "|", "/"])
        self.loading = False

    def spinner_task(self):
        while self.loading:
            sys.stdout.write(next(self.spinner_signs))  # write the next character
            sys.stdout.flush()  # flush stdout buffer (actual character display)
            time.sleep(self.delay)
            sys.stdout.write("\b")  # erase the last written char

    def start(self, model):
        printy(f"Loading model: {model}")
        self.loading = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.loading = False
        time.sleep(self.delay)
        sys.stdout.write("\b \b")  # erase the spinner and leave the cursor at the start


# Now we modify the usage of the BlenderbotPipeline to include the loading animation

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
