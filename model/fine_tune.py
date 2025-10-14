"""Fine-tuning script for VEAH LLM with Solana data"""

from peft import LoraConfig, get_peft_model
import torch

def fine_tune(base_model, dataset_path="data/solana_docs"):
    """Fine-tune model with LoRA on Solana data"""

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Fine-tuning logic here
    return model