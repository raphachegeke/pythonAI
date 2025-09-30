import torch
from transformers import ViTImageProcessor, ViTForImageClassification, VisionEncoderDecoderModel, GPT2TokenizerFast, AutoProcessor
from PIL import Image
import speech_recognition as sr
import pyttsx3
import os

# --- 1. INITIALIZATION: Setting up the AI's Senses and Voice ---

# The "Voice" (Text-to-Speech Engine)
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing text-to-speech engine: {e}")
    exit()

# The "Ears" (Speech Recognizer)
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# The "Eyes" (Image Captioning Model)
# We use a pre-trained model from Hugging Face that combines a Vision Transformer (ViT) and a GPT-2 language model.
print("Loading AI vision model... (This may take a moment)")
try:
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the AI model: {e}")
    print("Please ensure you have an internet connection and enough disk space.")
    exit()

# --- 2. CORE FUNCTIONS: What the AI can do ---

def speak(text):
    """Converts text to speech."""
    print(f"AI: {text}")
    engine.say(text)
    engine.runAndWait()

def listen_for_command():
    """Listens for a voice command and returns it as text."""
    with microphone as source:
        print("\nAdjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening for a command...")
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing...")
            command = recognizer.recognize_google(audio) # Using Google for better accuracy
            print(f"You said: {command}")
            return command.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return ""

def describe_image(image_path):
    """Analyzes an image and returns a textual description."""
    if not os.path.exists(image_path):
        return f"Error: The file '{image_path}' was not found."
        
    try:
        # Open and process the image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert(mode='RGB')

        # The model expects images in a specific format
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        # Generate the caption
        generated_ids = model.generate(pixel_values, max_length=50)
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption
    except Exception as e:
        return f"An error occurred while processing the image: {e}"

# --- 3. MAIN LOOP: The AI's Consciousness ---

if __name__ == "__main__":
    IMAGE_FILE = "image_to_see.jpg" # Make sure this image is in the same folder
    
    speak("Hello. I am an AI that can see and hear. Say 'describe the image' to begin.")
    
    while True:
        command = listen_for_command()
        
        if "describe" in command and "image" in command:
            speak("Okay, I am looking at the image now.")
            description = describe_image(IMAGE_FILE)
            speak(description)
        elif "exit" in command or "quit" in command or "stop" in command:
            speak("Goodbye!")
            break
        elif command: # If some other command was understood
            speak("I'm not sure how to respond to that. Please say 'describe the image'.")
