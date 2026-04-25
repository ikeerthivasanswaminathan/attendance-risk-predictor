import os
from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# List all available models
print("\n📌 Available Gemini Models:\n")

for m in client.models.list():
    print(m.name)