import random
import datasets
from template.protocol import Challenge
from template.validator.tasks import TASKS

def get_synapse():
    # task_name = random.choice(TASKS)
    task_name = 'abercrombie'
    
    dataset = datasets.load_dataset("nguha/legalbench", task_name)
    test_df = dataset["test"].to_pandas()
    test_df = test_df.drop(columns=['index'])
    
    rand_entry = test_df.sample(n=1)
    output = rand_entry['answer'].values[0]
    input = rand_entry.drop(columns=['answer']).to_dict(orient='records')[0]
    
    return Challenge(task_type=task_name, problem=input), output

if __name__ == '__main__':
    get_synapse()