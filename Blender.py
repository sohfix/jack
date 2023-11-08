from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
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
