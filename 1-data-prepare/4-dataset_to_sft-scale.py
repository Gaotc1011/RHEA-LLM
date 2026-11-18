import json
import re


def convert_jsonl(input_path, tasks, time):
         
    prev_composition = None
    prev_density = None
    converted_data = []

    for task in tasks:
        with open(input_path, 'r', encoding='utf-8') as infile: 
            for line in infile:# 提取属性字段
                
                original = json.loads(line.strip())

                composition_str = original["composition_formula"]
                process_str = original["process_description"]

                properties = {task: original.get(task,"N")}
                
                # 检查properties中是否有值为'N' 0的字段
                if any(value == 'N' for value in properties.values()):
                    continue  # 跳过该条目
                

                task_cleaned = re.sub(r'\(.*?\)', '', task).strip().split("_")
                task_cleaned = " ".join(task_cleaned)
                # print(task_cleaned)

                if any(float(value) == 0 for value in properties.values()):
                    continue  # 跳过该条目
                if "strain" in task:
                    if float(properties[task]) >=50:
                        continue    
                if "density" in task:
                    if prev_composition == composition_str and float(prev_density) == float(properties[task]):# 判断是否重复
                        continue

                property = {re.sub(r'\(.*?\)', '', task).strip(): str(round(float(value) * time, 3)) for task, value in properties.items()}
                property_str = json.dumps(property, ensure_ascii=False)

                new_entry = {
                    "instruction": f"Predict the {task_cleaned} for the following refractory high entropy alloy.",
                    "input": f"{composition_str} alloy {process_str}",
                    "output": f"{property_str}"
                }
                converted_data.append(new_entry)

                prev_composition = composition_str
                prev_density = properties[task]
                # print(len(converted_data))
            
    return converted_data



tasks = ["compressive_yield_strength_at_room_temperature(MPa)", "compressive_strain_at_room_temperature(%)", 
             "compressive_yield_strength_at_1073K(MPa)", "compressive_yield_strength_at_1273K(MPa)", "hardness(HV)"]
times = [0.1, 1,10,100]

for time in times:
    converted_data_train = convert_jsonl('./extraction-output/alloy_database-all.jsonl', tasks, time)
    path = f"./sft_data/train_data-{time}.json"
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(converted_data_train, outfile, indent=4, ensure_ascii=False)

    print(f"Data Converted with length(train): {len(converted_data_train)}")

