import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score

# Load GPT-2 model for synthetic data generation
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

# Load the tokenizer and model for classification
tokenizer_cls = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_cls = AutoModelForSequenceClassification.from_pretrained("./results")

# Generate synthetic samples
def generate_synthetic_samples(prompt, num_samples=5, max_length=50):
    inputs = tokenizer_gpt2(prompt, return_tensors="pt")
    samples = []
    for _ in range(num_samples):
        outputs = model_gpt2.generate(**inputs, max_length=max_length, do_sample=True)
        samples.append(tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True))
    return samples

# Tokenize the input text
def tokenize_data(data):
    return tokenizer_cls(data, return_tensors="pt", padding=True, truncation=True)

# Classify the data
def classify_data(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).tolist()

# Log hate speech queries
def log_hate_speech(data, predictions):
    with open("hate_speech_log.txt", "w") as log_file:
        for i, (text, pred) in enumerate(zip(data, predictions)):
            if pred == 1:
                log_file.write(f"Query {i}: {text}\n")

# Calculate recall and precision
def calculate_metrics(predictions, labels):
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return precision, recall

# Main function
def main():
    # Generate synthetic hate speech and non-hate speech data
    hate_speech_prompt = "Hate speech example: "
    non_hate_speech_prompt = "Friendly speech example: "
    
    hate_speech_data = generate_synthetic_samples(hate_speech_prompt, num_samples=50)
    non_hate_speech_data = generate_synthetic_samples(non_hate_speech_prompt, num_samples=50)
    
    data = hate_speech_data + non_hate_speech_data
    labels = [1] * len(hate_speech_data) + [0] * len(non_hate_speech_data)
    
    # Tokenize the data
    inputs = tokenize_data(data)
    
    # Classify the data
    predictions = classify_data(model_cls, inputs)
    
    # Log hate speech queries
    log_hate_speech(data, predictions)
    
    # Calculate recall and precision
    precision, recall = calculate_metrics(predictions, labels)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

if __name__ == "__main__":
    main()
