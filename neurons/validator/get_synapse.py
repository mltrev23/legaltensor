import os
import requests
import random
import time
import json
import logging
import asyncio
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from neurons.validator.tasks import TASKS

from template.protocol import Challenge

load_dotenv()

def get_synapse_from_server():
    try:
        data_server_url = os.environ.get('DATA_SERVER_URL')
        response = requests.get(data_server_url)
        data = response.json()
        return Challenge(task_type=data['task_type'], problem=data['input']), data.output
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

async def generate_synapse_using_openai():
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    task_name = random.choice(TASKS)
    print(f'Task Name: {task_name}')
    
    system_prompt = open(f'./legalbench/tasks/{task_name}/README.md').read()
    
    test_df = pd.read_csv(f'./legalbench/tasks/{task_name}/train.tsv', sep='\t')
    test_df = test_df.drop(columns=['index'])
    
    rand_entry = test_df.sample(n=1)
    output = rand_entry['answer'].values[0]
    input = rand_entry.drop(columns=['answer']).to_dict(orient='records')[0]
    
    user_prompt = f"Q: {input}\nA: {output}"
        
    response = await client.chat.completions.create(
        model = 'gpt-4',
        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': user_prompt
        }]
    )
    print(response)
    input = response.split('Q:')[1].split('A:')[0]
    output = response.split('A:')[1]
    return Challenge(task_type=task_name, problem=json.load(input)), output
    
def get_synapse():
    attempts = 3
    while attempts > 0:
        synapse = get_synapse_from_server()
        if synapse:
            return synapse
        logging.info(f'Failed to get synapse from server. Attempts {4 - attempts}...')
        retry_in = (4 - attempts) * 200 + 500
        logging.info(f'Retrying in {retry_in} milliseconds...')
        time.sleep(retry_in / 1000)
        attempts -= 1

    synapse = asyncio.run(generate_synapse_using_openai())
    if synapse:
        return synapse
if __name__ == '__main__':
    asyncio.run(generate_synapse_using_openai())