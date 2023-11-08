from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
)

import threading
import itertools
import time
import sys
from printy import inputy, printy

MODELS = {
    "B": { 1: "facebook/blenderbot-1B-distill", 9: "hyunwoongko/blenderbot-9B", 3: 'facebook/blenderbot-3B'},
    "M": {400: 'facebook/blenderbot-400M-distill'},
}


class BlenderbotPipeline:
    def __init__(self, model_name):
        # Load the model and tokenizer
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    def __call__(self, text):
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt")

        # Generate a response
        reply_ids = self.model.generate(**inputs)

        # Decode the model output into a human-readable format
        reply = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        return reply


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
