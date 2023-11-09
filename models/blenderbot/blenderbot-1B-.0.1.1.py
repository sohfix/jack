import datetime
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
version = '0.1.1'

class ManagedConversation:
    def __init__(self, model_name):
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.context = ""

    def add_user_input(self, user_input):
        # Update context
        self.context += "\nUser: " + user_input + "\n"

    def generate_response(self):
        # Generate response from model
        inputs = self.tokenizer([self.context], return_tensors='pt', padding=True)
        reply_ids = self.model.generate(**inputs)
        response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Update context with model response
        self.context += "Bot: " + response + "\n"

        return response


class ChatLogger:
    @staticmethod
    def log_conversation(conversation):
        with open("chat_history.txt", "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"{timestamp}: Blenderbot-1B {version} - Conversation:\n{conversation}\n\n"
            file.write(entry)


# Initialize the ManagedConversation with the Blenderbot model
model_name = "facebook/blenderbot-1B-distill"
conversation = ManagedConversation(model_name)

# Start a conversation
while True:
    if input('Proceed? [y/N] >') == 'N':
        break
    user_input = input("Enter your message: ")
    conversation.add_user_input(user_input)
    response = conversation.generate_response()
    print(f"{model_name}:", response)

    # After each interaction, log the conversation
    ChatLogger.log_conversation(conversation.context)
