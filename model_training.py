## Data set source: https://huggingface.co/datasets/odegiber/hate_speech18/tree/main
## Data Fields
# text: the provided sentence
# user_id: information to make it possible to re-build the conversations these sentences belong to
# subforum_id: information to make it possible to re-build the conversations these sentences belong to
# num_contexts: number of previous posts the annotator had to read before making a decision over the category of the sentence
# label: hate, noHate, relation (sentence in the post doesn't contain hate speech on their own, but combination of serveral sentences does) or idk/skip (sentences that are not written in English or that don't contain information as to be classified into hate or noHate)
#%%
from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_dataset("odegiber/hate_speech18")

# Filter the dataset to include only the first two classes
def filter_classes(example):
    return example["label"] in [0, 1]

filtered_dataset = dataset.filter(filter_classes)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True)

# Split the dataset
train_test = tokenized_datasets["train"].train_test_split(test_size=0.1)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# %%
