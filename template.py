def preprocess_slimorca_mistral(x):

    # 결과 문자열 구성
    if next((item for item in x if item['from'] == 'system'), {'value': ''})['value']:    
        return f"[INST] ### System: {next((item for item in x if item['from'] == 'system'), {'value': 'No System Message'})['value']} \
                ### User: {next((item for item in x if item['from'] == 'human'), {'value': 'No User Message'})['value']} [/INST] \
                ### Assistant: {next((item for item in x if item['from'] == 'gpt'), {'value': 'No Assistant Message'})['value']}"
    else:
        return f"[INST] ### User: {next((item for item in x if item['from'] == 'human'), {'value': 'No User Message'})['value']} [/INST] \
                ### Assistant: {next((item for item in x if item['from'] == 'gpt'), {'value': 'No Assistant Message'})['value']}"


def preprocess_alpaca_mistral(row):
    if row['input'] == '':
        return f"[INST] ### System: Below is an instruction that describes a task. Write a response that appropriately completes the request.\
                #### User: {row['instruction']} [/INST]  \
                ### Assistant: {row['output']}"
    else:
        return f"[INST] ### System: Below is an instruction that describes a task. Write a response that appropriately completes the request.\
                ### User: {row['instruction']} \
                ### Input: {row['input']} [/INST] \
                ### Assistant: {row['output']}"
    
def preprocess_orcamath_mistral(row):
        return f"[INST] ### Input: {row['question']} [/INST] \
                ### Assistant: {row['answer']}"


def preprocess_slimorca_gemma(x):
    
    if next((item for item in x if item['from'] == 'system'), {'value': ''})['value']:
        return f"### System: {next((item for item in x if item['from'] == 'system'), {'value': 'No System Message'})['value']} \
                ### User: {next((item for item in x if item['from'] == 'human'), {'value': 'No User Message'})['value']} \
                ### Assistant: {next((item for item in x if item['from'] == 'gpt'), {'value': 'No Assistant Message'})['value']}"
    else:
        return f"### User: {next((item for item in x if item['from'] == 'human'), {'value': 'No User Message'})['value']} \
                ### Assistant: {next((item for item in x if item['from'] == 'gpt'), {'value': 'No Assistant Message'})['value']}"


def preprocess_alpaca_gemma(row):
    if row['input'] == '':
        return f"### System: Below is an instruction that describes a task. Write a response that appropriately completes the request.\
                #### User: {row['instruction']} \
                ### Assistant: {row['output']}"
    else:
        return f"### System: Below is an instruction that describes a task. Write a response that appropriately completes the request.\
                ### User: {row['instruction']} \
                ### Input: {row['input']} \
                ### Assistant: {row['output']}"  

def preprocess_orcamath_gemma(row):
      return f"### Input: {row['question']} \
                ### Assistant: {row['answer']}"