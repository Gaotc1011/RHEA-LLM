import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
import json
import os

# 配置参数
MODEL_PATH = "./LLM_model\Qwen3-8B"  # 原始本地模型路径
FINETUNED_PATH = "./Model_train\All-Qwen-8B-finetuned-full-20250717-152459\checkpoint-644"  # 微调后的模型路径
TEST_DATA_PATH = "../data_extraction/sft_data/train_data-20250707-all.json"  # 测试集路径
TRAINING_METHOD = "full"  # 与训练时一致
OUTPUT_FOLDER = "./multitask_results"
OUTPUT_PATH = f"{OUTPUT_FOLDER}/train_predictions_Qwen3-8B-finetuned-{TRAINING_METHOD}-all-644.jsonl"

def load_finetuned_model():
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

def format_prompt(example):
    """格式化指令为模型输入"""

    if example.get('input'):
        return f"<|im_start|>system\nYou are a materials science expert.<|im_end|>\n<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>system\nYou are a materials science expert.<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"

def generate_response(model, tokenizer, prompt):
    """生成模型回复"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成配置
    generation_config = GenerationConfig(
        max_new_tokens=64,
        do_sample=True,
        repetition_penalty=1, 
        temperature=0.6, 
        top_p=0.9, 
        top_k=50,
        # no_repeat_ngram_size=2,  # 防止短语重复
        remove_invalid_values=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
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
    return clean_response

def evaluate_test_data():
    # 1. 加载模型
    model, tokenizer = load_finetuned_model()
    
    # 2. 加载测试数据
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        # test_data = json.load(f)[:5]  # 只取前3条
    
    # 3. 评估每条数据
    results = []
    for i, example in enumerate(test_data, 1):
        print(f"\n{'='*50}")
        print(f"测试样本 {i}/{len(test_data)}")
        
        # 获取真实回复
        true_response = example['output']
        
        # 格式化输入
        prompt = format_prompt(example)
        # 生成模型回复
        model_response = generate_response(model, tokenizer, prompt)
        
        # 保存结果
        results.append({
            "instruction": example['instruction'],
            "input": example.get('input', ""),
            "output": true_response,
            "predict": model_response
        })
        
        print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    print("开始评估微调模型在测试集上的表现...")
    evaluation_results = evaluate_test_data()
    
    # 保存评估结果
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in evaluation_results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n') 
        # json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估完成! 结果已保存至: {OUTPUT_PATH}")