import os
import requests
import random
import json
import torch
import pandas as pd
import bittensor as bt
from openai import OpenAI
from dotenv import load_dotenv
from neurons.validator.tasks import TASKS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from template.protocol import Challenge

load_dotenv()

task_creation_prompt = """
Please create a new task and sample output using the specified format below.
Ensure that the question (Q) is in JSON format and the answer (A) provides a clear and concise response.

Example here: 
{examples}
"""

MODELS = dict()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def get_synapse_from_server():
    try:
        data_server_url = os.environ.get('DATA_SERVER_URL')
        response = requests.get(data_server_url)
        data = response.json()
        return Challenge(task_type=data['task_type'], problem=data['input']), data.output
    except Exception as e:
        bt.logging.error(f"An error occurred: {e}")
        return None

def generate_prompts(task_name):
    system_prompt = open(f'./legalbench/tasks/{task_name}/README.md').read()
    
    test_df = pd.read_csv(f'./legalbench/tasks/{task_name}/train.tsv', sep='\t')
    test_df = test_df.drop(columns=['index'])
    
    rand_entry = test_df.sample(n=1)
    output = rand_entry['answer'].values[0]
    input = rand_entry.drop(columns=['answer']).iloc[0].to_json()
    
    user_prompt = f"Q: {input}\nA: {output}"
    
    return system_prompt, user_prompt

def generate_chat_completion_message(system_prompt, user_prompt):
    return [{
        'role': 'system',
        'content': system_prompt
    }, {
        'role': 'user',
        'content': task_creation_prompt.format(examples = user_prompt)
    }]

def generate_synapse_using_openai():
    task_name = random.choice(TASKS)
    
    system_prompt, user_prompt = generate_prompts(task_name)
    chat_completion_message = generate_chat_completion_message(system_prompt, user_prompt)
    
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
    response = client.chat.completions.create(
        model = 'gpt-4',
        messages = chat_completion_message
    )
    response = response.choices[0].message.content
    
    input = response.split('Q: ')[1].split('A: ')[0]
    output = response.split('A: ')[1]
    
    return Challenge(task_type=task_name, problem=json.loads(input)), output

def load_llama(device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    model_id = "cognitivecomputations/dolphin-2.9.4-llama3.1-8b"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,           # This flag is now part of BitsAndBytesConfig
        bnb_4bit_use_double_quant=True,  # Optional, for double quantization
        bnb_4bit_quant_type="nf4",   # Choose between 'fp4' or 'nf4' (Non-negative quantization)
    )

    if not hasattr(AutoModelForCausalLM, 'cached_model'):
        AutoModelForCausalLM.cached_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,  # 4-bit Quantization config
            torch_dtype=torch.bfloat16,        # Mixed precision (optional, use bfloat16 for efficiency)
        ).to(device)
    model = AutoModelForCausalLM.cached_model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def generate_synapse_using_llama():
    task_name = random.choice(TASKS)
    
    system_prompt, user_prompt = generate_prompts(task_name)
    chat_completion_message = generate_chat_completion_message(system_prompt, user_prompt)
    
    # If model already exists ...
    if 'llama' not in MODELS:
        MODELS['llama'] = load_llama(device)
    model, tokenizer = MODELS['llama']
    
    def preprocess_message(messages):
        text = [f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>" for message in messages]
        text = "\n".join(text)
        return f'{text.strip()}<|im_start|>assistant'

    message = preprocess_message(chat_completion_message)

    # Prepare the input question
    input_ids = tokenizer.encode(message, return_tensors="pt").to(device)

    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=400, num_return_sequences = 1).to(device)

    # Decode the generated answer
    output_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_answer = output_answer.split("<|im_start|>assistant")[1].strip()
    
    input = output_answer.split('Q: ')[1].split('A: ')[0]
    output = output_answer.split('A: ')[1]

    return Challenge(task_type=task_name, problem=json.loads(input)), output

synapse_generation_methods = {
    generate_synapse_using_openai: 0.5,
    generate_synapse_using_llama: 0.5,
}

def function_selector(function_ratios):
    total = sum(function_ratios.values())
    cumulative = []
    current_sum = 0

    for func, ratio in function_ratios.items():
        current_sum += ratio
        cumulative.append((current_sum / total, func))

    # Generate a random number and select the corresponding function
    rand = random.random()
    for threshold, func in cumulative:
        if rand <= threshold:
            return func

def get_synapse():
    # attempts = 3
    # while attempts > 0:
    #     synapse = get_synapse_from_server()
    #     if synapse:
    #         return synapse
    #     logging.info(f'Failed to get synapse from server. Attempts {4 - attempts}...')
    #     retry_in = (4 - attempts) * 200 + 500
    #     logging.info(f'Retrying in {retry_in} milliseconds...')
    #     time.sleep(retry_in / 1000)
    #     attempts -= 1

    func = function_selector(synapse_generation_methods)
    synapse = func()
    bt.logging.info(f'Synapse generated: {synapse}')
    
    return synapse
