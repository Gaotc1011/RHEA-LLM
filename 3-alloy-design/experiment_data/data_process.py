import json
from collections import defaultdict
import re
import csv
from sklearn.model_selection import train_test_split


with open("D:\gaotianchuang\property_prediction_by_deepseek\data_extraction\output/alloy_database-all.jsonl", 'r', encoding='utf-8') as f:
    data =  [json.loads(line.strip()) for line in f]

reject_rocess = ["roll", "press", "HIP", "anneal", "homogenization", "age"]    
refect_element = ["B","C","N","O","Co","Fe","Ni","La","Re","Y"]
# property_list = ["density(g/cm³)",  "compressive_yield_strength_at_room_temperature(MPa)", "compressive_yield_strength_at_1073K(MPa)"]
# property_list = ["hardness(HV)",  "compressive_yield_strength_at_1073K(MPa)", "compressive_yield_strength_at_1273K(MPa)"]
# property_list = ["density(g/cm³)",  "compressive_yield_strength_at_room_temperature(MPa)", "compressive_strain_at_room_temperature(%)"]
property_list = ["hardness(HV)",  "compressive_strain_at_room_temperature(%)", "compressive_yield_strength_at_1273K(MPa)"]

with open('experiment_data_VAM_4.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        if item['process_method_1'] == 'arc melting':
            # if iin type in item['process_method_2'],then continue
            if any(x in item['process_method_2'] for x in reject_rocess):
                continue
            elements = []
            matches = re.findall(r'([A-Z][a-z]?)(\d+\.?\d*|\d*\.\d+)', item['composition_formula'])
            for element, conc in matches:
                elements.append(element)
            if any(x in elements for x in refect_element):
                continue

            if all(item.get(prop, 'N') != 'N' for prop in property_list):
                if 'compressive_strain_at_room_temperature(%)' in property_list:
                    strain_value = float(item['compressive_strain_at_room_temperature(%)'])
                    if strain_value >= 50:
                        continue  # strain >= 50 时不保存

                output_item = {
                    'composition_formula': item['composition_formula'],
                    **{prop: item[prop] for prop in property_list}
                }

                json_record = json.dumps(output_item, ensure_ascii=False)
                f.write(json_record + '\n')

