import torch
from PIL import Image
from transformers import AutoTokenizer
import random
import time
import pandas as pd
import json
from rouge import Rouge

# Setting a random seed for reproducibility
random.seed(42)

# Function to load CSV data
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    findings = df['findings'].tolist()
    impressions = df['impressions'].tolist()
    image_paths = [path.strip().split(",")[0] for path in df['images'].tolist()]
    return findings, impressions, image_paths

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f_json:
        return json.load(f_json)

# Function to calculate average ROUGE scores
def calculate_average_rouge_scores(generated_results, actual_impressions_dict, limit=None):
    rouge = Rouge()
    scores = []
    for i, (image_path, generated_impression) in enumerate(generated_results.items()):
        if limit and i >= limit:
            break
        actual_impression = actual_impressions_dict.get(image_path)
        if actual_impression:
            score = rouge.get_scores(generated_impression, actual_impression)
            scores.append(score)
        else:
            print(f"No actual impression found for image path: {image_path}")

    # Calculating averages
    average_scores = {metric: {'f': [], 'p': [], 'r': []} for metric in ['rouge-1', 'rouge-2', 'rouge-l']}
    for score in scores:
        for metric in average_scores:
            for measure in average_scores[metric]:
                average_scores[metric][measure].append(score[0][metric][measure])

    for metric in average_scores:
        for measure in average_scores[metric]:
            average_scores[metric][measure] = sum(average_scores[metric][measure]) / len(average_scores[metric][measure])

    return average_scores

# Main execution
if __name__ == "__main__":
    # Loading data
    findings, impressions, image_paths = load_csv_data('/home/choonghan/practice/test_full.csv')
    generated_results = load_json_data('/home/choonghan/practice/results_test_retriever_80_findings.json')
    actual_impressions_dict = dict(zip(image_paths, impressions))

    # Calculating ROUGE scores
    average_scores = calculate_average_rouge_scores(generated_results, actual_impressions_dict, limit=1000)
    print("Average ROUGE scores:", average_scores)
