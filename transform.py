from transformers import pipeline
from printy import printy, inputy


class TextModels:
    def __init__(
        self,
        task="text-generation",
        model="gpt2-medium",
    ):
        self.task = task
        self.model = model
        self.pipe = None
        self.prompt = None
        self.queue = []
        self.gen_text_tot = []

    def init_pipe(self):
        self.pipe = pipeline(self.task, model=self.model)

    def load_prompts(self, d: list):
        _ = [self.queue.insert(0, i) for i in d]

    @staticmethod
    def create_prompts(a=True, n=5):
        if a:
            return [inputy("Enter a prompt: ", "c") for i in range(n)]
        else:
            return False

    def choose_by_index(self, index):
        return self.queue.pop(index)

    # Define the function to pretty print the list and return a string based on the chosen index
    def print_list_and_choose(self):
        # Pretty print each element with its index
        for index, element in enumerate(self.queue):
            printy(f"{index}: {element}", "y")

        chosen_index = int(
            inputy(
                "Enter the [B]index@ of the word you choose: ",
                "g",
            )
        )

        return self.choose_by_index(chosen_index)


from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Conversation,
)


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

    """# Usage
# Initialize the custom conversational AI pipeline
custom_pipe = BlenderbotPipeline("facebook/blenderbot-1B-distill")

# Generate a response using the custom pipeline
response = custom_pipe("Hello, how are you?")

# Print the generated response
print(response)"""
