from printy import inputy
from Blender import LoadingBar, MODELS, ManagedConversation

# Usage
loading_bar = LoadingBar()
prefix = 'B'
size = 1

# todo count tokens, time thinking
model = MODELS[prefix][size]

loading_bar.start(model)
conversation = ManagedConversation(model)
loading_bar.stop()

# Loop for conversing with the model
for i in range(5):
    user_input = 'eat a bunch of shit' # inputy(f"[c]Welcome to the {model}, ask away: @")
    conversation.add_user_input(user_input)
    response = conversation.generate_response()
    print(response)
