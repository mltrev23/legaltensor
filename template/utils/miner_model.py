# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datasets

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Equall/Saul-7B-Instruct-v1")
model = AutoModelForCausalLM.from_pretrained("Equall/Saul-7B-Instruct-v1").to(device)

def process(prompt: str):
    # Prepare the input for the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate a response
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1)

    # Decode the generated output
    return tokenizer.decode(output[0], skip_special_tokens=True)
