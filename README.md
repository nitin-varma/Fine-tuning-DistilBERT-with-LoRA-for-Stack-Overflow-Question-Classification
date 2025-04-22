# Fine-tuning DistilBERT with LoRA for Stack Overflow Question Classification

This project demonstrates how to fine-tune a DistilBERT model using Low-Rank Adaptation (LoRA) for the task of classifying Stack Overflow questions as either "open" or "closed". The fine-tuned model can help identify questions that are likely to be closed due to being off-topic, non-constructive, or otherwise unsuitable for the Stack Overflow platform.

## Project Overview

Stack Overflow faces challenges with question quality. This project implements a machine learning solution to automatically classify questions, helping to:
- Reduce moderation workload
- Improve content quality
- Provide feedback to users before posting

We fine-tune DistilBERT using LoRA, a parameter-efficient transfer learning technique that significantly reduces the number of trainable parameters while maintaining performance.

## Requirements
torch>=1.8.0
transformers>=4.18.0
datasets>=2.0.0
peft>=0.3.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
tqdm>=4.60.0

## Setup Instructions

1. Clone this repository:
   git clone <repo link>
   cd stackoverflow-question-classifier
2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Download the dataset:
   The Stack Overflow dataset is available in the `data` directory. If not present, download it from [Kaggle](https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data).


## Usage

### Training

To fine-tune the model, run the Jupyter notebook:
jupyter notebook Notebook(1).ipynb

The notebook contains all steps from data preprocessing to model evaluation and error analysis.

### Inference

You can use the fine-tuned model for prediction with:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load the tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Load the fine-tuned LoRA model
merged_model = PeftModel.from_pretrained(base_model, "models/distilbert_fine_tuned_lora/merged")

# Function for prediction
def predict_question_status(question_text):
    inputs = tokenizer(question_text, padding=True, truncation=True, return_tensors="pt")
    outputs = merged_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    return "open" if prediction == 0 else "closed", probs[0][prediction].item()

# Example
status, confidence = predict_question_status("How do I sort a list in Python?")
print(f"Question status: {status} (confidence: {confidence:.2f})")

## Results

The fine-tuned model achieves 77.65% accuracy on the test set, with a macro F1 score of 0.776. The model performs better on "open" questions (F1=0.82) than on "closed" questions (F1=0.73).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* HuggingFace Transformers and PEFT libraries
* Stack Overflow for the dataset
