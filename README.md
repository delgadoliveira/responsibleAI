# Responsible AI Framework: Hate Speech Classification POC

## Overview

This project aims to develop a Proof of Concept (POC) for a responsible AI framework, specifically focusing on the classification of hate speech. The goal is to detect and mitigate instances of hate speech in AI-generated content, ensuring that it aligns with ethical standards and promotes a safe, inclusive digital environment.

## Table of Contents

- [Responsible AI Framework: Hate Speech Classification POC](#responsible-ai-framework-hate-speech-classification-poc)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Purpose](#purpose)
  - [Methodology](#methodology)
    - [Data Preparation](#data-preparation)
    - [Model Training](#model-training)
    - [Synthetic Data Generation](#synthetic-data-generation)
    - [Classification and Logging](#classification-and-logging)
  - [Components](#components)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Purpose

The primary purpose of this POC is to:
- Detect and mitigate hate speech in AI-generated content.
- Promote fairness and inclusivity by filtering content related to race, gender, and other sensitive topics.
- Enhance trust in AI systems by addressing abuse patterns and ensuring responsible AI usage.

## Methodology

### Data Preparation
1. **Dataset**: We utilize the `hate_speech18` dataset available on Hugging Face, which contains labeled data for hate speech and non-hate speech.
2. **Filtering**: The dataset is filtered to include only the first two classes (labels 0 and 1), ensuring that the model is trained for binary classification.
3. **Tokenization**: The dataset is tokenized using the `distilbert-base-uncased` tokenizer, transforming text data into a format suitable for model training.

### Model Training
1. **Model**: A `distilbert-base-uncased` model is fine-tuned for sequence classification with two labels (0 for non-hate speech and 1 for hate speech).
2. **Training**: The model is trained using the `Trainer` API from Hugging Face, with appropriate training arguments such as batch size, learning rate, and number of epochs.
3. **Evaluation**: The model is evaluated on a validation set, and metrics such as accuracy, precision, and recall are recorded.

### Synthetic Data Generation
1. **Model**: For generating synthetic data, we use the `Phi-2` model from Microsoft Research, which is efficient and capable of generating high-quality text.
2. **Prompts**: Synthetic hate speech and non-hate speech data are generated using specific prompts, ensuring diversity in the generated samples.
3. **Tokenization**: Generated synthetic data is tokenized using the `distilbert-base-uncased` tokenizer for classification.

### Classification and Logging
1. **Classification**: The tokenized synthetic data is classified using the trained `distilbert-base-uncased` model.
2. **Logging**: Queries classified as hate speech are logged for further review and analysis.
3. **Metrics**: Precision and recall metrics are calculated to evaluate the performance of the classification model on synthetic data.

## Components

1. **Content Classification**: Utilizes classifier models to detect harmful language related to hate speech in user prompts and outputs.
2. **Human Review and Decision**: Flags and logs hate speech content for authorized personnel to review and confirm the classification based on predefined guidelines.
3. **Content Filtering System**: Works with core models to detect and prevent harmful content in both input prompts and 
## Installation

To get started with the project, follow these steps:

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/responsible-ai-framework.git
```

2. Navigate to the project directory:
```bash
   cd responsibleAI
```

3. Install the required dependences:
```bash
   pip install -r requirements.txt   
```

## Usage

1. Run the hate speech classification model:
```bash
   python classify_hate_speech.py   
```

2. Review flagged content:

* Access the logs to review and confirm or correct classifications based on predefined guidelines.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.

2. Create a new branch (git checkout -b feature-branch).

3. Make your changes and commit them (git commit -m 'Add new feature').

4. Push to the branch (git push origin feature-branch).

5. Create a Pull Request.

## License 

This project is licensed under the MIT license. See the [LICENSE](https://mit-license.org/) file for details.
