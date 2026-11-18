import json
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split



def convert_jsonl(input_path, tasks, random_state):
    converted_data_train = []
    converted_data_val = []
         
    for  task, state in zip(tasks, random_state):
        prev_composition = None
        prev_density = None
        converted_data = []
        with open(input_path, 'r', encoding='utf-8') as infile: 
            for line in infile:# 提取属性字段
                
                original = json.loads(line.strip())

                composition_str = original["composition_formula"]
                process_str = original["process_description"]

                properties = {task: original.get(task,"N")}
                
                # 检查properties中是否有值为'N' 0的字段
                if any(value == 'N' for value in properties.values()):
                    continue  # 跳过该条目

                property_str = json.dumps(properties, ensure_ascii=False)

                task_cleaned = re.sub(r'\(.*?\)', '', task).strip().split("_")
                task_cleaned = " ".join(task_cleaned)
                # print(task_cleaned)

                if "phase" in task_cleaned:
                        new_entry = {
                        "instruction": f"Determine whether the given refractory high entropy alloy has {task_cleaned}.",
                        "input":f"{composition_str} alloy {process_str}" ,
                        "output": f"{property_str}"
                        }
                        converted_data.append(new_entry)                  
                
                else: 
                    if any(float(value) == 0 for value in properties.values()):
                        continue  # 跳过该条目
                    if "strain" in task:
                        if float(properties[task]) >=50:
                            continue    
                    if "density" in task:
                        if prev_composition == composition_str and float(prev_density) == float(properties[task]):
                            continue

                    new_entry = {
                        "instruction": f"Predict the {task_cleaned} for the following refractory high entropy alloy.",
                        "input": f"{composition_str} alloy {process_str}",
                        "output": f"{property_str}"
                    }
                    converted_data.append(new_entry)

                    prev_composition = composition_str
                    prev_density = properties[task]
            # print(len(converted_data))
        
        data_train, data_val = train_test_split(converted_data, test_size=0.2, random_state=state)
        # data_train= data_val = converted_data
        converted_data_train.extend(data_train)
        converted_data_val.extend(data_val)
            
    return converted_data_train, converted_data_val




task_list = ["density(g/cm³)", "hardness(HV)", "compressive_yield_strength_at_room_temperature(MPa)", "compressive_strain_at_room_temperature(%)", 
             "compressive_yield_strength_at_1073K(MPa)", "compressive_yield_strength_at_1273K(MPa)","single_solution_phase", "intermetallic_phase"]
random = [49,49,49,49,20,49,49,49]
# task_list = ["intermetallic_phase"]
converted_data_train, converted_data_val = convert_jsonl('./extraction-output/alloy_database-all.jsonl', tasks = task_list, random_state = random)

with open('./sft_data/train_data-all.json', 'w', encoding='utf-8') as outfile:
    json.dump(converted_data_train, outfile, indent=4, ensure_ascii=False)

with open('./sft_data/val_data-all.json', 'w', encoding='utf-8') as outfile:
    json.dump(converted_data_val, outfile, indent=4, ensure_ascii=False)

print(f"Data Converted with length(train): {len(converted_data_train)}, length(val): {len(converted_data_val)}")