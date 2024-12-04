# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Equall/Saul-7B-Instruct-v1")
model = AutoModelForCausalLM.from_pretrained("Equall/Saul-7B-Instruct-v1").to(device)

def process(prompt: str):
    # Prepare the input for the model
    tokenizer.pad_token_id = tokenizer.eos_token_id  # or specify your own
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate a response
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)

    # Decode the generated output
    return tokenizer.decode(output[0], skip_special_tokens=True)
