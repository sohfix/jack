import datetime
from transformers import GPT2Tokenizer, OPTForCausalLM
from utils.speak import TextToSpeechPlayer

version = "0.0.1"


class ManagedConversation:
    def __init__(self, model_name, max_context_tokens=128):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = OPTForCausalLM.from_pretrained(model_name)
        self.context = ""
        self.max_context_tokens = max_context_tokens

    def add_user_input(self, user_input):
        self.context += "\nUser: " + user_input + "\n"
        self.trim_context()

    def generate_response(self):
        # Encode the inputs
        inputs = self.tokenizer.encode(
            self.context, return_tensors="pt", add_special_tokens=True
        )

        # Generate response using the encoded inputs
        reply_ids = self.model.generate(
            input_ids=inputs, max_length=self.max_context_tokens + len(inputs[0])
        )

        # Decode the generated response
        response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Append the bot's response to the context
        self.context += "Bot: " + response + "\n"
        self.trim_context()

        return response

    def trim_context(self):
        tokens = self.tokenizer.tokenize(self.context)
        while len(tokens) > self.max_context_tokens:
            # Remove the first line (oldest message) to shorten the context
            first_newline_index = self.context.find("\n")
            self.context = self.context[first_newline_index + 1 :]
            tokens = self.tokenizer.tokenize(self.context)  # Re-tokenize after trimming


class ChatLogger:
    @staticmethod
    def log_conversation(conversation):
        with open("chat_history.txt", "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = (
                f"{timestamp}: OPT-1.3B {version} - Conversation:\n{conversation}\n\n"
            )
            file.write(entry)


# Initialize the ManagedConversation with the OPT model
model_name, name_i = "facebook/opt-1.3b", "Jack the cat bot"
conversation = ManagedConversation(model_name)
robot = TextToSpeechPlayer()

# Start a conversation
while True:
    if input("Proceed? [y/N] > ") == "N":
        break
    user_input = input("Enter your message: ")
    conversation.add_user_input(user_input)
    response = conversation.generate_response()
    print(f"{model_name}:", response)
    robot.play_text(f"{name_i} says: {response}")
    # After each interaction, log the conversation
    ChatLogger.log_conversation(conversation.context)
