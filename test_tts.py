import torchaudio
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

wav = model.generate("Hello! Chatterbox is working on GPU.")
torchaudio.save("test_output.wav", wav, model.sr)
print("Saved test_output.wav")
