from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
)
import torch
import threading
import itertools
import time
import sys
from printy import inputy, printy

MODELS = {
    "B": {
        1: "facebook/blenderbot-1B-distill",
        9: "hyunwoongko/blenderbot-9B",
        3: "facebook/blenderbot-3B",
    },
    "M": {400: "facebook/blenderbot-400M-distill"},
}

CUDA = f"CUDA avail={torch.cuda.is_available()}\nGPU={torch.cuda.get_device_name(0)}"


class BlenderbotPipeline:
    def __init__(self, model_name):
        # Load the model and tokenizer
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    def __call__(
        self,
        text,
        max_length=200,
        min_length=100,
        length_penalty=1.5,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
    ):
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt")

        # Generate a response with the specified parameters for longer responses
        reply_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
        )

        # Decode the model output into a human-readable format
        reply = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        return reply


class LoadingBar:
    def __init__(self, delay=0.1):
        self.delay = delay  # In seconds
        self.spinner_signs = itertools.cycle(["--", "\\", "||", "//"])
        self.loading = False

    def spinner_task(self):
        while self.loading:
            sys.stdout.write(next(self.spinner_signs))  # write the next character
            sys.stdout.flush()  # flush stdout buffer (actual character display)
            time.sleep(self.delay)
            sys.stdout.write("\b")  # erase the last written char

    def start(self, text='loading'):
        printy(f"{text} > from memory...")
        self.loading = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.loading = False
        time.sleep(self.delay)
        sys.stdout.write("\b \b")  # erase the spinner and leave the cursor at the start


class ManagedConversation:
    def __init__(self, model):
        self.history = []
        self.pipe = BlenderbotPipeline(model)

    def add_user_input(self, text):
        self.history.append(text)

        if len(self.history) > 3:
            load = LoadingBar()
            load.start("Trimming")
            try:
                self.trim()
            finally:
                load.stop()

    def trim(self, index=3):
        while len(self.history) > index:
            self.history.pop(0)

    # TODO implement way to keep conversations below 128

    def generate_response(self):
        # Join the conversation history into a single string
        context = " ".join(self.history)
        # Generate the response
        response = self.pipe(context)
        # Add the generated response to the history
        self.history.append(response)
        return response
