import os
import requests
import random
import json
import pandas as pd
import bittensor as bt
from openai import OpenAI
from dotenv import load_dotenv
from neurons.validator.tasks import TASKS

from template.protocol import Challenge

load_dotenv()

task_creation_prompt = """
Please create a new task using the specified format below.
Ensure that the question (Q) is in JSON format and the answer (A) provides a clear and concise response.

{examples}
"""

def get_synapse_from_server():
    try:
        data_server_url = os.environ.get('DATA_SERVER_URL')
        response = requests.get(data_server_url)
        data = response.json()
        return Challenge(task_type=data['task_type'], problem=data['input']), data.output
    except Exception as e:
        print(f"An error occurred: {e}")
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

def generate_synapse_using_openai():
    task_name = random.choice(TASKS)
    print(f'Task Name: {task_name}')
    
    system_prompt, user_prompt = generate_prompts(task_name)
    
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
    response = client.chat.completions.create(
        model = 'gpt-4',
        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': task_creation_prompt.format(examples = user_prompt)
        }]
    )
    response = response.choices[0].message.content
    
    input = response.split('Q: ')[1].split('A: ')[0]
    output = response.split('A: ')[1]
    
    return Challenge(task_type=task_name, problem=json.loads(input)), output

def generate_synapse_using_llama():
    
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

    synapse = generate_synapse_using_openai()
    if synapse:
        return synapse