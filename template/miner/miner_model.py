# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import datasets

pipe = pipeline("text-generation", model="Equall/Saul-Instruct-v1", torch_dtype=torch.bfloat16, device_map="auto")

def process(prompt: str):
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
    full_output = outputs[0]["generated_text"]

    # Remove the input prompt from the output
    generated_text = full_output[len(prompt):].strip()

    # Print the generated response
    return generated_text