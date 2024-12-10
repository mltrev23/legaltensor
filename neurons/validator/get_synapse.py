import os
import requests

from template.protocol import Challenge

def get_synapse():
    try:
        data_server_url = os.environ.get('DATA_SERVER_URL')
        response = requests.get(data_server_url)
        data = response.json()
        return Challenge(task_type=data.task_type, problem=data.input), data.output
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    get_synapse()