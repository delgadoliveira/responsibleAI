#%%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import precision_score, recall_score

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

# Filter and tokenize the dataset
def filter_classes(example):
    return example["label"] in [0, 1]

# Main function
def main():
    # Load the pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    # Sample texts
    texts = [
        "I love everyone!",
        "I hate you!"
    ]

    # Classify each text
    for text in texts:
        classification = classify_text(model, tokenizer, text)
        print(f"Text: {text}\nClassification: {classification}\n")
    

    # Load the dataset
    dataset = load_dataset("odegiber/hate_speech18")

    filtered_dataset = dataset.filter(filter_classes)

    # Extract the texts and labels
    texts = filtered_dataset['train']['text']
    labels = filtered_dataset['train']['label']

    # Classify and evaluate
    predictions = [classify_text(model, tokenizer, text) for text in texts]

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

if __name__ == "__main__":
    main()








# %%
