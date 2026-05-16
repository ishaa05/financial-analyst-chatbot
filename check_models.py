# check_models.py
from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(api_version="v1beta")
)

for m in client.models.list():
    if "generate" in str(m.supported_actions).lower():
        print(m.name)