from transformers import pipeline
import json
import re

# Define categories and their keywords for classification
categories = {
    "What": ["checks", "verifies", "determines", "retrieves", "validates", "finds", "computes", "calculates"],
    "Why": ["reason", "purpose", "rationale", "explains", "aims", "intended", "because"],
    "How-to-use": ["usage", "set-up", "initialize", "call", "expected", "precondition", "usage example"],
    "How-it-is-done": ["performs", "does", "executes", "handles", "implements", "processes", "creates", "manages"],
    "Property": ["returns", "asserts", "checks", "validates", "condition", "pre-condition", "post-condition"],
    "Others": ["done", "finished", "cleanup", "free", "resources"]
}

# Load the pre-trained BART model for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to classify a comment using keyword matching
def classify_with_keywords(comment):
    # Check if any category matches the comment based on predefined keywords
    for category, keywords in categories.items():
        if any(keyword.lower() in comment.lower() for keyword in keywords):
            return category
    return None  # Return None if no match found

# Function to classify a comment using BERT (zero-shot classification)
def classify_with_bert(comment):
    result = classifier(comment, candidate_labels=list(categories.keys()))
    return result['labels'][0]

# Function to update the labels in the dataset
def update_labels(dataset):
    for entry in dataset:
        comment = entry["comment"]
        
        # First try to classify using keyword-based rules
        predicted_label = classify_with_keywords(comment)
        
        # If no label is found using keywords, use BERT to classify
        if predicted_label is None:
            predicted_label = classify_with_bert(comment)
        
        entry["label"] = predicted_label  # Assign the predicted label
    return dataset

# Load the dataset from a JSON file
INPUT_FILE = f"ruby_dataset.json"
input_file = f'ruby_dataset/{INPUT_FILE}'  # Replace with your input JSON file path
with open(input_file, 'r', encoding='utf-8') as json_file:
    dataset = json.load(json_file)

# Update the dataset with correct labels
updated_dataset = update_labels(dataset)

# Print the updated dataset (optional)
for entry in updated_dataset:
    print(f"ID: {entry['id']}, Label: {entry['label']}, Comment: {entry['comment']}")

# Save the updated dataset to a new JSON file
output_file = f'labelled_data/labelled_{INPUT_FILE}'  # The output JSON file with labels
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(updated_dataset, json_file, indent=4)

print(f"Dataset labeled and saved to '{output_file}'.")