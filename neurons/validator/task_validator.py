import requests
import socket
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import pandas as pd
import numpy as np

from legaltensor.legaltensor.neurons.llms.saul_lm import process

def get_ip_address():
    try:
        # Get the hostname
        hostname = socket.gethostname()
        # Get the IP address
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        return f"Error: {e}"
    
# Scoring Engine
class ScoringEngine:
    def __init__(self, vector_db_endpoint):
        self.vector_db_endpoint = vector_db_endpoint
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_uniqueness(self, task_prompt, top_k = 10):
        task_vector = self.embedder.encode(task_prompt)
        task_embeddings = requests.get(self.vector_db_endpoint).json()
        similarities = [cosine_similarity(task_vector, embedding) for embedding in task_embeddings]
        top_similar_embeddings = similarities.sort()[top_k]
        similarity = sum(top_similar_embeddings) / top_k

        return 1 - similarity

    def calculate_completeness(self, readme_md, prompt_txt, train_tsv, test_tsv, threshold=0.8):
        readme_length = len(readme_md.split())
        prompt_length = len(prompt_txt.split())

        readme_length_score = max(0.25, readme_length / 2000)
        prompt_length_score = max(0.25, prompt_length / 1200)

        train_data_score = self.calculate_data_accuracy(train_tsv, readme_md, prompt_txt)
        test_data_score = self.calculate_data_accuracy(test_tsv, readme_md, prompt_txt)

        if train_data_score < threshold:
            return -1
        if test_data_score < threshold:
            return -1

        data_accuracy_score = (train_data_score + test_data_score) / 4

        return readme_length_score + prompt_length_score + data_accuracy_score

    def calculate_data_accuracy(self, tsv_file_data, readme_md, prompt_txt, k = 10):
        tsv_data = StringIO(tsv_file_data)
        df = pd.read_csv(tsv_data, sep='\t')

        sampled_data = df.sample(n=k)
        scores = []
        for i in range(k):
            entry = sampled_data.iloc[i].to_dict()
            prompt = readme_md + prompt_txt.format(**entry)

            result = process(prompt)
            score = cosine_similarity(np.array(self.embedder.embed(result), np.array(self.embedder.embed(entry['answer']))))
            scores.append(score)
        
        return sum(scores) / k

    def calculate_bonus(self, readme_md):
        return 1 if "special" in readme_md.lower() else 0.0


class TaskValidatorServer:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.scoring_engine = ScoringEngine(vector_db)
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/score_task")
        async def score_task(task_data: dict):
            task_id = task_data['task_id']
            metadata = task_data['metadata']

            uniqueness = self.scoring_engine.calculate_uniqueness(metadata['readme_md'])
            completeness = self.scoring_engine.calculate_completeness(metadata['readme_md'], metadata['prompt_txt'])
            bonus = self.scoring_engine.calculate_bonus(metadata['readme_md'])

            total_score = uniqueness * 0.4 + completeness * 0.4 + bonus * 0.2

            # response = {
            #     "validator_id": hash(self),
            #     "task_id": task_id,
            #     "uniqueness": uniqueness,
            #     "completeness": completeness,
            #     "bonus": bonus,
            #     "total": total_score
            # }

            # requests.post("http://contribution-api.local/receive_scores", json=response)
            return total_score
        
    def run(self, port = 20502):
        import uvicorn
        uvicorn.run(self.app, host='0.0.0.0', port=port)
        ip_addr = get_ip_address()
        task_approval_api_url = f'http://{ip_addr}:{port}'