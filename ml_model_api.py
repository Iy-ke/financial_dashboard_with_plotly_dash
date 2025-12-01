import os
import requests

# Your Hugging Face API token is read from an environment variable
API_TOKEN = os.getenv('HF_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

def query_sentiment_model(text_input: str) -> dict | None:
    """Queries the Hugging Face sentiment model API and returns the result."""
    if not API_TOKEN:
        print("ERROR: Hugging Face API token (HF_TOKEN) not set.")
        return None

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": text_input}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()[0]
        top_result = max(result, key=lambda x: x['score'])
        return top_result
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None





















