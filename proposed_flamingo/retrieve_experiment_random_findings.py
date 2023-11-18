import torch
from PIL import Image
from transformers import AutoTokenizer
import random
import time
import pandas as pd
import json
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

# Set a random seed for reproducibility
random.seed(42)

# Read and prepare the data
base_path = '/home/choonghan/practice/'
test_data_file = 'data_test_corrupted_0.8.csv'
train_data_file = 'data_train_corrupted_0.8.csv'
json_path = 'results_test_retriever_80_findings.json'

def prepare_data(file_name):
    df = pd.read_csv(f'{base_path}{file_name}')
    findings = df['findings'].tolist()
    impressions = df['impressions'].tolist()  
    image_paths = [f"{base_path}{path.strip().split(',')[0]}" for path in df['images'].tolist()]
    return findings, impressions, image_paths

test_findings, test_impressions, test_image_paths = prepare_data(test_data_file)
train_findings, train_impressions, train_image_paths = prepare_data(train_data_file)

# Initialize the model and tokenizer
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1
)

# Load the model checkpoint
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to(device)

# Attempt to load existing results or initialize a new dictionary
try:
    with open(f'{base_path}{json_path}', "r") as f_json:
        results_dict = json.load(f_json)
except FileNotFoundError:
    results_dict = {}

# Define constants
max_tokens_findings = 350
max_tokens_impressions = 350
max_length = 2045

# Process each item in the dataset
for i in range(len(test_findings)):
    # Compose the new image path
    image_path_new = test_image_paths[i].replace(base_path, "")

    # Skip if the result already exists
    if image_path_new in results_dict:
        print(f"Skipping generation for {image_path_new} as it already exists in the JSON file.")
        continue

    try:
        # Sample random indices for training data
        random_indices = random.sample(range(len(train_findings)), 2)

        # Create the prompt with findings and impressions
        prompt = 'Be a radiologist and write "impression", which is a summarization of findings of a radiology image. These are example impressions.'

        for idx in random_indices:
            prompt += create_prompt_entry(train_findings[idx], train_impressions[idx], tokenizer, max_tokens_findings, max_tokens_impressions)

        # Add the current test finding to the prompt
        prompt += create_prompt_entry(test_findings[i], "", tokenizer, max_tokens_findings, max_tokens_impressions, add_impression=False)

        # Tokenize and truncate the prompt
        tokens_prompt = tokenizer.encode(prompt, add_special_tokens=False)[:max_length]
        prompt = tokenizer.decode(tokens_prompt).strip()

        # Load and process images
        images = [Image.open(path.strip()) for path in [train_image_paths[idx] for idx in random_indices] + [test_image_paths[i]]]
        vision_x = torch.cat([image_processor(img).unsqueeze(0) for img in images], dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)

        # Tokenize the prompt for language model
        lang_x = tokenizer([prompt], return_tensors="pt", padding_side="left")
        lang_x = {k: v.to(device) for k, v in lang_x.items()}

        # Generate the impression
        start_time = time.time()
        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=300,
            num_beams=3,
        )
        elapsed_time = time.time() - start_time
        result = tokenizer.decode(generated_text[0], skip_special_tokens=True)[len(prompt):].strip()
        
               # Log the generation result and time taken
        print(f"Generated text: {result}")
        print(f"Time taken for generation: {elapsed_time:.2f} seconds")

        # Save the result in the dictionary
        results_dict[image_path_new] = result

    except AssertionError as error:
        print(f"Skipping index {i} due to error: {error}")
        continue

    # Save the results to a JSON file periodically
    if i % 10 == 0 or i == len(test_findings) - 1:
        with open(f'{base_path}{json_path}', "w") as f_json:
            json.dump(results_dict, f_json, indent=4)

def create_prompt_entry(finding, impression, tokenizer, max_tokens_findings, max_tokens_impressions, add_impression=True):
    """
    Creates a prompt entry from the given findings and impressions.
    """
    finding_str = finding.strip()
    tokens_findings = tokenizer.encode(finding_str, add_special_tokens=False)
    if len(tokens_findings) > max_tokens_findings:
        tokens_findings = tokens_findings[:max_tokens_findings]
    finding_str = tokenizer.decode(tokens_findings).strip()

    prompt_entry = f'<image>Findings: {finding_str}'

    if add_impression:
        impression_str = impression.strip()
        tokens_impressions = tokenizer.encode(impression_str, add_special_tokens=False)
        if len(tokens_impressions) > max_tokens_impressions:
            tokens_impressions = tokens_impressions[:max_tokens_impressions]
        impression_str = tokenizer.decode(tokens_impressions).strip()
        prompt_entry += f'.Impression:{impression_str}'

    return prompt_entry

# Main processing loop
if __name__ == "__main__":
    for i in range(len(test_findings)):
        # Processing each item in the test dataset
        process_test_item(i)

