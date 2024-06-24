import ktrain
import requests
from dotenv import load_dotenv
import os

load_dotenv()

input_text = "Conor is probably the biggest MMA fighter."

print("=== Text classification from trained model ===")
p = ktrain.load_predictor('./model/text_classifier')
print(p.predict(input_text))

print("=== Text classification from pre-trained model ===")
print("===  IAB Content Taxonomy V3 ===")
iab = requests.post('https://api.uclassify.com/v1/uclassify/iab-content-taxonomy-v3/classify',
                    json={'texts': [input_text]}, headers={'Authorization': f'Token {os.getenv("UCLASSIFY_TOKEN")}'})
print(iab.json())

print("=== Topics ===")
topics = requests.post('https://api.uclassify.com/v1/uclassify/topics/classify',
                       json={'texts': [input_text]}, headers={'Authorization': f'Token {os.getenv("UCLASSIFY_TOKEN")}'})
print(topics.json())
