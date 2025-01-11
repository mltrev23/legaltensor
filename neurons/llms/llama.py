from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import torch

def load_llama(device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    model_id = "cognitivecomputations/dolphin-2.9.4-llama3.1-8b"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,           # This flag is now part of BitsAndBytesConfig
        bnb_4bit_use_double_quant=True,  # Optional, for double quantization
        bnb_4bit_quant_type="nf4",   # Choose between 'fp4' or 'nf4' (Non-negative quantization)
    )

    if not hasattr(AutoModelForCausalLM, 'cached_model'):
        AutoModelForCausalLM.cached_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,  # 4-bit Quantization config
            torch_dtype=torch.bfloat16,        # Mixed precision (optional, use bfloat16 for efficiency)
        ).to(device)
    model = AutoModelForCausalLM.cached_model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

model, tokenizer = load_llama()

def process(prompt: str, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Process a list of messages.

    Args:
        messages (list): A list of message objects to process.

    Returns:
        The processed result.
    """
    message = f"<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant"

    # Prepare the input question
    input_ids = tokenizer.encode(message, return_tensors="pt").to(device)

    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=400, num_return_sequences = 1).to(device)

    # Decode the generated answer
    output_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_answer = output_answer.split("<|im_start|>assistant")[1].strip()

    return output_answer