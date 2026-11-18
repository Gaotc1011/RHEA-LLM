import swanlab
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import os
import json
from datasets import Dataset
import accelerate
import time

# swanlab.login(api_key="Your api key", save=True) # run it at first time with your api key


# 配置参数
MODEL_PATH = "./LLM_model\Qwen3-8B"
TRAIN_DATASET_PATH = "../data-prepare/sft_data/train_data-All.json"  # 替换为你的数据集路径
VAL_DATASET_PATH = "../data-prepare/sft_data/val_data-All.json"
TRAINING_METHOD = "full"  # 可选: "full", "freeze", "lora"
OUTPUT_DIR = f"All-Qwen-8B-finetuned-{TRAINING_METHOD}-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
EPOCHS = 4
BATCH_SIZE = 2  # 根据GPU显存调整
MAX_LENGTH = 1024


swanlab.config.update({
    "model": MODEL_PATH,
    "train_method": TRAINING_METHOD,
    "output_dir": OUTPUT_DIR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "test_dataset_path": VAL_DATASET_PATH,
    "max_length": MAX_LENGTH,
    })



def load_model_and_tokenizer():
    """加载模型和tokenizer（不使用量化）"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    
    # model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    return model, tokenizer

def format_instruction(example):
    """格式化指令数据"""
    if example.get('input'):
        prompt = f"<|im_start|>system\nYou are a materials science expert.<|im_end|>\n<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>system\nYou are a materials science expert.<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    
    # 只计算assistant回复的loss
    full_text = prompt + example['output'] + "<|im_end|>"
    
    return {"text": full_text, "prompt": prompt}


def tokenize_function(examples):
    """Tokenize函数"""
    tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding=False)
    instruction = tokenizer(examples["prompt"], truncation=True, max_length=MAX_LENGTH, padding=False)
    
    # 创建labels - 只计算assistant回复的loss
    labels = []
    for i in range(len(instruction["input_ids"])):
        input_ids = tokenized_inputs["input_ids"][i]
        prompt_len = len(instruction ["input_ids"][i])
        
        # 创建labels，prompt部分设为-100（忽略loss）
        label = [-100] *prompt_len + input_ids[prompt_len:]
        labels.append(label)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_dataset(data_path):
    """准备训练数据集"""
    # 加载JSON格式数据集
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 格式化指令
    formatted_data = [format_instruction(ex) for ex in dataset]
    
    # 转换为数据集对象
    dataset = Dataset.from_dict({
        "text": [ex["text"] for ex in formatted_data],
        "prompt": [ex["prompt"] for ex in formatted_data]
    })
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "prompt"]
    )
    return tokenized_dataset


def prepare_model_for_training(model, method="lora"):
    """根据训练方法准备模型
    参数:
        method (str): 训练方法 ("full", "freeze", "lora")
        unfreeze_layers (int): 解冻最后几层的层数（例如：2 表示最后两层）"""
    if method == "lora":
        # 准备LoRA训练
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
       
    elif method == "freeze":
        unfreeze_layers = 4
        # 获取所有层的数量
        total_layers = model.config.num_hidden_layers

        # 动态生成要保留的层名称
        trainable_layer_names = [
            f"model.layers.{total_layers - i}" for i in range(1, unfreeze_layers + 1)
        ] + ["lm_head"]

        # 冻结除指定层外的所有参数
        for name, param in model.named_parameters():
            if not any(trainable_name in name for trainable_name in trainable_layer_names):
                param.requires_grad = False

        # model.print_trainable_parameters()
        total_params = sum(p.numel() for p in model.parameters())    
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"All parameters: {total_params}, Trainable parameters: {trainable_params}, percent: {100 * trainable_params / total_params:.2f}%")
    
    # full方法不需要特殊处理
    return model


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """训练模型"""
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",#"cosine",constant
        weight_decay=0,
        warmup_ratio=0.05,
        optim="adamw_torch",
        per_device_eval_batch_size=BATCH_SIZE,
        eval_steps=100,
        eval_strategy="steps",  
        logging_steps=20,
        save_steps=161,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="swanlab",
        run_name=f"All-Qwen-8B-{TRAINING_METHOD}"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    trainer.train()

    return model



if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 准备数据集
    train_dataset = prepare_dataset(TRAIN_DATASET_PATH)
    val_dataset = prepare_dataset(VAL_DATASET_PATH)
    print(f"train_dataset: {len(train_dataset)}")
    print(train_dataset[0])
    # print(len(train_dataset[0]["input_ids"]))
    # exit(0)
  
    # 3. 根据训练方法准备模型
    model = prepare_model_for_training(model, TRAINING_METHOD)
    
    # 4. 训练模型
    trained_model = train_model(model, tokenizer, train_dataset, val_dataset)
    print(f"训练完成! 模型已保存到: {OUTPUT_DIR}")

    swanlab.finish()







