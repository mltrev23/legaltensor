def read_prompts_from_file(task_type):
    with open(f'./legalbench/tasks/{task_type}/base_prompt.txt', 'r') as f:
        return str(f.read())
    return None
