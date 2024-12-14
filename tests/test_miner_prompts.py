from neurons.miner.miner_prompts import read_prompts_from_file
from neurons.validator.tasks import TASKS
import random

def test_read_prompts():
    result = read_prompts_from_file(random.choice(TASKS))
    assert result is not None
    
    print(result)
