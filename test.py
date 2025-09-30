# import torch
# import transformers
# import PIL
# import speech_recognition as sr
# import pyttsx3
# import pyaudio
# import importlib.metadata

# print("✅ Torch version:", torch.__version__)
# print("✅ Transformers version:", transformers.__version__)
# print("✅ Pillow version:", PIL.__version__)
# print("✅ SpeechRecognition version:", sr.__version__)

# # pyttsx3 doesn't have __version__, so use importlib.metadata
# print("✅ pyttsx3 version:", importlib.metadata.version("pyttsx3"))
# print("✅ PyAudio version:", pyaudio.__version__)

# # Try listing audio devices
# pa = pyaudio.PyAudio()
# print("\n🔊 Available audio devices:")
# for i in range(pa.get_device_count()):
#     dev = pa.get_device_info_by_index(i)
#     print(f" - {i}: {dev['name']} (inputs: {dev['maxInputChannels']}, outputs: {dev['maxOutputChannels']})")
# pa.terminate()

# print("\n✅ All libraries imported successfully!")


import speech_recognition as sr
import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()

# Say something
engine.say("Hello! Your local setup is working.")
engine.runAndWait()

# Record from mic
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("🎤 Say something...")
    audio = recognizer.listen(source, timeout=5)

try:
    text = recognizer.recognize_google(audio)
    print("✅ You said:", text)
except Exception as e:
    print("❌ Error:", e)
