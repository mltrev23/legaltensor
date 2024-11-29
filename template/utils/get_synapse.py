import random
import datasets
from template.protocol import Challenge
from template.utils.tasks import TASKS

def get_synapse():
    # task_name = random.choice(TASKS)
    task_name = 'abercrombie'
    
    dataset = datasets.load_dataset("nguha/legalbench", task_name)
    test_df = dataset["test"].to_pandas()
    
    rand_entry = random.choice(test_df.values)
    
    return Challenge(task_type=task_name, problem=rand_entry[2]), rand_entry[0]