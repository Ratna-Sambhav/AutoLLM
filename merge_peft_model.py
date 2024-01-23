from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def merge_peft_adaptors(base_model_name, peft_model_dir, new_model_name, new_model_dir):

    # Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Loading adapter: {args.peft_model}
    model = PeftModel.from_pretrained(base_model, peft_model_dir, device_map="auto")
    
    # Merge base model and adapter
    model = model.merge_and_unload()
    
    # Saving model and tokenizer in {args.hub_id}
    model_path = os.path.join(new_model_name, new_model_dir)
    model.save_pretrained(f"{model_path}")
    tokenizer.save_pretrained(f"{model_path}")

if __name__ == "__main__" :
    pass

# Usage:
# python merge_peft.py --base_model=meta-llama/Llama-2-7b-hf --peft_model=./qlora-out --new_model_name=Llama-2-7b-fine-tuned --model_save_dir=./final_model_weights