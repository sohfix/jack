from printy import inputy, printy
from Blender import LoadingBar, MODELS, ManagedConversation

# Usage
loading_bar = LoadingBar()
prefix = "B"
size = 1

# todo count tokens, time thinking
model = MODELS[prefix][size]

loading_bar.start(model)
conversation = ManagedConversation(model)
loading_bar.stop()

# TODO start logging all conversations!!!!!

# Loop for conversing with the model
for i in range(2):
    user_input = (
        "Write a note from a ship in 1860."
        # input(f"[c]Welcome to the {model}, ask away: @")
    )
    conversation.add_user_input(user_input)
    response = conversation.generate_response()
    printy(response, "Bc")
