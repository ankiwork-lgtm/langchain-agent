from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available Embedding Models:")
print("-" * 50)

try:
    models = client.models.list()
    
    for model in models:
        # Check if model supports embeddings
        if hasattr(model, 'supported_generation_methods'):
            if 'embedContent' in model.supported_generation_methods or 'embed' in model.name.lower():
                print(f"✓ {model.name}")
                if hasattr(model, 'description'):
                    print(f"  Description: {model.description}")
                print()
except Exception as e:
    print(f"Error listing models: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: Just list common embedding models
    common_models = [
        "models/embedding-001",
        "models/text-embedding-004",
        "embedding-001",
        "text-embedding-004"
    ]
    
    print("\nCommon embedding models to try:")
    for model in common_models:
        print(f"  - {model}")