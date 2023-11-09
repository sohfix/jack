from gtts import gTTS
from playsound import playsound
import os
import uuid


class TextToSpeechPlayer:
    def __init__(self, language='en', slow=False):
        self.language = language
        self.slow = slow

    def play_text(self, text):
        # Generate a unique filename for the temporary audio file.
        temp_file = f"temp_audio_{uuid.uuid4().hex}.mp3"

        try:
            # Convert the text to speech.
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            # Save the audio file.
            tts.save(temp_file)
            # Play the audio file.
            playsound(temp_file)
        finally:
            # Clean up: remove the audio file if it exists.
            if os.path.isfile(temp_file):
                os.remove(temp_file)


# Example usage:
if __name__ == "__main__":
    player = TextToSpeechPlayer()
    player.play_text('Hello, how are you doing today?')
