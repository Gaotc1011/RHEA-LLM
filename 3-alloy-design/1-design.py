import json
from openai import OpenAI
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
import re
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

config = {
        
        "model_path": "./LLM_model/Qwen3-8B",
        "finetuned_model_path": "....",
        "reproduction_size": 50000,

        }

## Initial dataset loading
def rule_based_generation(reproduction_size, epoch=1):
    elements = ['Nb','Ti','V','Zr','Ta','Mo','Al','Hf','Cr','W','Si']
    element_order = {e:i for i,e in enumerate(elements)}
    candidates = []
    max_attempts = 10  # Prevent infinite loops
    
    for _ in range(reproduction_size):
        valid = False
        attempts = 0
        
        while not valid and attempts < max_attempts:
            attempts += 1
            # Randomly select 4-7 elements
            n_elements = np.random.randint(4, 8)
            selected_elements = np.random.choice(elements, n_elements, replace=False)
            
            # Generate random fractions (5-35%) that sum to 100
            # Generate random fractions (5-35%) that sum to 100
            fractions = np.zeros(n_elements)

            for i in range(n_elements):
                elem = selected_elements[i]
                if elem in ['Nb', 'Ti', 'V', 'Zr']:
                    # Group A: 5~35
                    fractions[i] = np.random.uniform(5, 40)
                elif elem in ['Ta', 'Mo', 'Al', 'Hf', 'Cr', 'W']:
                    # Group B: 5~30
                    fractions[i] = np.random.uniform(5, 25)
                elif elem == 'Si':
                    # Group C: 0~10
                    fractions[i] = np.random.uniform(0, 10)

            fractions = np.round(fractions / fractions.sum() * 100, 1)
            
            # Ensure last element makes total 100%
            fractions[-1] = 100 - np.sum(fractions[:-1])
            
            # Validate each element based on its group
            valid = True
            for i in range(n_elements):
                elem = selected_elements[i]
                val = fractions[i]

                if elem in ['Nb', 'Ti', 'V', 'Zr'] and not (5 <= val <= 40):
                    valid = False
                    break
                elif elem in ['Ta', 'Mo', 'Al', 'Hf', 'Cr', 'W'] and not (5 <= val <= 25):
                    valid = False
                    break
                elif elem == 'Si' and not (0 <= val <= 10):
                    valid = False
                    break
            if valid:
                # Format composition string
                sorted_elements, sorted_fractions = zip(*sorted(zip(selected_elements, fractions), key=lambda x: element_order[x[0]]))
                composition = "".join(f"{e}{f:.1f}" for e,f in zip(sorted_elements, sorted_fractions))
                candidates.append({"iteration": epoch, "composition": composition})
            elif attempts == max_attempts:
                print(f"Warning: Couldn't generate valid composition after {max_attempts} attempts")
    
    return candidates


## Property prediction
def load_finetuned_model(TRAINING_METHOD, MODEL_PATH, FINETUNED_PATH):
    """加载微调后的模型"""
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH)
    
    if TRAINING_METHOD == "lora":
        print("加载LoRA微调模型...")
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, FINETUNED_PATH)
    else:
        print("加载全量微调模型...")
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    return model, tokenizer


def predict_property(input, model, tokenizer):
    # model, tokenizer = load_finetuned_model(TRAINING_METHOD, MODEL_PATH, FINETUNED_PATH)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(input, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=64,
        do_sample=True,
        repetition_penalty=1, 
        temperature=0.6, 
        top_p=0.9, 
        top_k=50,
        # no_repeat_ngram_size=2,  # 防止短语重复
        remove_invalid_values=True,
    )
    
    # 生成回复
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )
    
    # 解码并提取assistant回复
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_response = full_response.split("<|im_start|>assistant")[-1]
    
    # 清理特殊token
    clean_response = assistant_response.replace("<|im_end|>", "").strip()

    return float(next(iter(json.loads(clean_response).values())))


def format_prompt(example, property="density"):
    if property == 'density':
        instruction = "Predict the density for the following refractory high entropy alloy."
    elif property == 'YS-RT':
        instruction = "Predict the compressive yield strength at room temperature for the following refractory high entropy alloy."
    elif property == 'YS-1073':
        instruction = "Predict the compressive yield strength at 1073K for the following refractory high entropy alloy."
    elif property == 'hardness':
        instruction = "Predict the hardness for the following refractory high entropy alloy."
    elif property == 'strain':
        instruction = "Predict the strain at room temperature for the following refractory high entropy alloy."
    elif property == 'YS-1273':
        instruction = "Predict the compressive yield strength at 1273K for the following refractory high entropy alloy."
    
    process = "prepared by vacuum arc melting with ingots re-melted five times."

    prompt = f"<|im_start|>system\nYou are a materials science expert.<|im_end|>\n<|im_start|>user\n{instruction}\n{example} alloy {process}<|im_end|>\n<|im_start|>assistant\n"

    return prompt


def property_prediction(example, model, tokenizer):
    valid_alloys = []
    i = 0
    
    for alloy in example:
        i += 1
        print(f"Processing sample {i}/{len(example)}")
        try:
            composition = alloy["composition"]
            for target in ['density', 'hardness', 'YS-RT', 'strain', 'YS-1073', 'YS-1273']:
                alloy[target] = predict_property(format_prompt(composition, property=target), model, tokenizer)

            if alloy['YS-1273'] <= alloy['YS-1073']:
                valid_alloys.append(alloy)
        except Exception as e:
            print(f"Unexpected error processing {composition}: {str(e)}")
            continue

    return valid_alloys


def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def main():
    population = rule_based_generation(reproduction_size = 5000)
    print(population)
    print(len(population))
    # exit(0)
    
    model, tokenizer = load_finetuned_model("full", config["model_path"], config["finetuned_model_path"])
    property = property_prediction(population, model, tokenizer)

    # save the result of all_materials
    save_jsonl(property, 'all_materials.jsonl')


if __name__ == "__main__":
    main()
