from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import precision_score, recall_score

# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Define the classification function
def classify_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Mapping logic: 0-2 -> 0 (hate speech), 3-4 -> 1 (non-hate speech)
    if predicted_class <= 2:
        return 0  # hate speech
    else:
        return 1  # non-hate speech

# Sample texts
texts = [
    "I love everyone!",
    "I hate you!"
]

# Classify each text and log results
log = []
for text in texts:
    classification = classify_text(model, tokenizer, text)
    mapping = "Hate Speech Detected" if classification == 0 else "No Hate Speech Detected"
    log.append({"text": text, "classification": mapping})
    print(f"Text: {text}\nClassification: {mapping}\n")

# Load the dataset
dataset = load_dataset("odegiber/hate_speech18")

# Filter and tokenize the dataset
def filter_classes(example):
    return example["label"] in [0, 1]

filtered_dataset = dataset.filter(filter_classes)
tokenized_datasets = filtered_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)

# Extract the texts and labels
texts = filtered_dataset['train']['text']
labels = filtered_dataset['train']['label']

# Classify and log results
predictions = []
log = []
for text, label in zip(texts, labels):
    classification = classify_text(model, tokenizer, text)
    mapping = "Hate Speech Detected" if classification == 0 else "No Hate Speech Detected"
    log.append({"text": text, "classification": mapping, "label": label})
    predictions.append(classification)

# Save the log to a file
with open("classification_log.txt", "w") as log_file:
    for entry in log:
        log_file.write(f"Text: {entry['text']}\nClassification: {entry['classification']}\nLabel: {entry['label']}\n\n")

# Calculate precision and recall
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
