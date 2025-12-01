# test_api.py
from ml_model_api import query_sentiment_model

print("Testing the ML model API...")
test_text = "The new product launch was a huge success."
result = query_sentiment_model(test_text)

if result:
    print("✅ Test successful!")
    print(f"Result: {result}")
else:
    print("❌ Test failed. Check your API token and the `ml_model_api.py` script.")