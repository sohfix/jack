from transformers import pipeline, AutoProcessor, AutoModel
import scipy.io.wavfile
import torch
import numpy as np
from mutil import ExecutionTimer
from printy import printy

class BarkTextToSpeech:
    def __init__(self, model_name="suno/bark-small", use_pipeline=True):
        self.model_name = model_name
        self.use_pipeline = use_pipeline
        if use_pipeline:
            self.synthesizer = pipeline("text-to-speech", model_name)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    def text_to_speech(self, text, output_file="output.wav", do_sample=True):
        if self.use_pipeline:
            speech = self.synthesizer(text, forward_params={"do_sample": do_sample})
            rate = speech["sampling_rate"]
            data = speech["audio"]
        else:
            inputs = self.processor(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            attention_mask = inputs.attention_mask
            with torch.no_grad():
                speech_values = self.model.generate(**inputs, attention_mask=attention_mask, do_sample=do_sample)
            rate = self.model.config.sample_rate
            data = speech_values.cpu().numpy().squeeze()

        # Convert to 16-bit PCM and clip the values
        data = np.int16(data / np.max(np.abs(data)) * 32767)
        data = np.clip(data, -32768, 32767)

        scipy.io.wavfile.write(output_file, rate, data)
        printy(f"Audio file saved as {output_file}", 'y')

# todo does not work
# Example usage
t = ExecutionTimer()
t.start()
bark_tts = BarkTextToSpeech()
bark_tts.text_to_speech("Hello, you are a jackass!", "bark-audio/my_audio.wav")
t.stop()
