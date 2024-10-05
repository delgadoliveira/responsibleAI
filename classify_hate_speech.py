import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./results")

# Define the input text
input_text = "Your input text here"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)
    
# Get the predicted label
predicted_label = torch.argmax(outputs.logits, dim=1).item()

# Interpret the results
label_names = ["Not Hate Speech", "Hate Speech"]
