import json
from collections import defaultdict
import re
import csv
from sklearn.model_selection import train_test_split


# 1. Read and parse JSONL file
def parse_jsonl(file_path):
    compositions = []
    alloy_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            alloy_data.append(data)
            if 'composition_formula' in data and data['composition_formula']:
                compositions.append(data['composition_formula'])
    return compositions, alloy_data


# 2. Extract elements and their concentrations from composition formulas
def extract_elements(compositions):
    all_element_data = []

    pattern = r'([A-Z][a-z]?)(\d+\.?\d*|\d*\.\d+)'  # Matches element symbols and concentrations

    for comp in compositions:
        matches = re.findall(pattern, comp)
        element_data = {}

        for element, conc in matches:
            try:
                concentration = float(conc)
                if concentration == 0:
                    continue
                element_data[element] = concentration
            except ValueError:
                continue

        all_element_data.append(element_data)

    return all_element_data


# 3. Enrich original JSONL data with element/concentration fields
def enrich_jsonl_with_elements(jsonl_file):
    compositions, alloy_data = parse_jsonl(jsonl_file)
    all_element_data = extract_elements(compositions)

    enriched_data = []

    for idx, (alloy, element_data) in enumerate(zip(alloy_data, all_element_data)):
        alloy['elements'] = element_data  # Add element data to each alloy entry
        enriched_data.append(alloy)

    return enriched_data


# 4. Convert enriched JSON data to CSV
def convert_to_csv(enriched_data, csv_file):
    fieldnames = set()

    # Collect all possible keys including nested element names
    for item in enriched_data:
        fieldnames.update(item.keys())
        if 'elements' in item:
            fieldnames.update(item['elements'].keys())

    fieldnames.discard('elements')  # Remove 'elements' key since it's expanded below
    fieldnames = ['Alloy_Index'] + sorted(list(fieldnames))  # Optional: sort or reorder fields

    with open(csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(enriched_data):
            row = {'Alloy_Index': idx}
            elements = item.pop('elements', {})
            row.update(item)
            row.update(elements)  # Flatten element concentrations as separate columns
            writer.writerow(row)

def data_process(data_path):
    task_list = ["density(g/cm³)", "hardness(HV)", "compressive_yield_strength_at_room_temperature(MPa)", "compressive_strain_at_room_temperature(%)", 
             "compressive_yield_strength_at_1073K(MPa)", "compressive_yield_strength_at_1273K(MPa)","single_solution_phase", "intermetallic_phase"]
    name_list = ["Density", "Hardness", "YS", "Strain", "YS1073K", "YS1273K", "SS", "IM"]
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data =  [json.loads(line.strip()) for line in f]

    for task in task_list:
        data_slip = []
        for item in data:
            if item[task] != 'N':
                data_slip.append(item)
        
        # slip the data in train and test set
        train_data, test_data = train_test_split(data_slip, test_size=0.2, random_state=42)
        


if __name__ == "__main__":
    jsonl_input_path = './data/data_VAM.jsonl'
    csv_output_path = './data/alloy_composition_data.csv'

    enriched_data = enrich_jsonl_with_elements(jsonl_input_path)
    convert_to_csv(enriched_data, csv_output_path)
