def read_prompts_from_file(task_type):
    with open(f'./base_prompts/{task_type}.txt', 'r') as f:
        return str(f.read())
    return None

if __name__ == '__main__':
    print(read_prompts_from_file('abercrombie'))