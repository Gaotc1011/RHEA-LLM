
# composition check
import json
import re
import numpy as np

# Function to read data from JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

# Function to parse composition formula into element-concentration pairs
def parse_composition(composition_formula):
    pattern = r'([A-Z][a-z]?)(\d+\.?\d*|\d*\.\d+)'
    matches = re.findall(pattern, composition_formula)
    return {element: float(conc) for element, conc in matches}

# Function to extract true and predicted elements and concentrations
def extract_elements_and_concentrations(true_data, pred_data):
    # 先收集所有出现过的元素
    all_known_elements = set()
    for entry in true_data + pred_data:
        elements = re.findall(r'([A-Z][a-z]?)\d', entry["composition_formula"])
        all_known_elements.update(elements)
    all_known_elements = sorted(all_known_elements)  # 固定顺序

    true_elements = []
    pred_elements = []
    true_concentrations = []
    pred_concentrations = []

    for true, pred in zip(true_data, pred_data):
        true_comp = parse_composition(true["composition_formula"])
        pred_comp = parse_composition(pred["composition_formula"])

        # 使用统一的元素集合
        true_elements.append([1 if elem in true_comp else 0 for elem in all_known_elements])
        pred_elements.append([1 if elem in pred_comp else 0 for elem in all_known_elements])

        true_concentrations.append([true_comp.get(elem, 0) for elem in all_known_elements])
        pred_concentrations.append([pred_comp.get(elem, 0) for elem in all_known_elements])
         # remove 0 data both in true_concentrations and pred_concentrations


    return (
        np.array(true_elements),
        np.array(pred_elements),
        np.array(true_concentrations),
        np.array(pred_concentrations)
    )

def composition_evaluate(LLM_data, manual_data, tolerance):
    results = {}
    TP, TN, FP, FN = 0, 0, 0, 0
    # Extract elements and concentrations
    true_elements, pred_elements, true_concentrations, pred_concentrations = extract_elements_and_concentrations(manual_data, LLM_data)

    for te, pe, tc, pc in zip(true_elements.flatten(), pred_elements.flatten(), true_concentrations.flatten(), pred_concentrations.flatten()):
        if te == 0:
            if pe == 0:
                TN += 1
                # continue
            else:
                FP += 1

        elif te == 1:
            if pe == 0:
                FN += 1
            elif pe == 1 and abs(tc - pc) <= tolerance:
                TP += 1
            else:
                FN += 1

    print(TP, FP, FN, TN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FN + FP)

    results["Composition"] = {"Recall": recall, "Precision": precision, "F1-score": f1, "Accuracy": accuracy}

    # Print results
    print(f"Composition Evaluation:  Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

    return results    


# Evaluation function for phase-related boolean fields
def phase_evaluate(LLM_data, manual_data):
    # Collect true and predicted values for both fields
    results = {}
    
    for phase in ["single_solution_phase", "intermetallic_phase"]:
        TP, TN, FP, FN = 0, 0, 0, 0
        i = 0
        for llm, manual in zip(LLM_data, manual_data):
            i += 1
        
            predict_value = str(llm[phase]).lower()
            true_value = str(manual[phase]).lower()
            
            if true_value == predict_value:
                if true_value == 'true':
                    TP += 1
                else:
                    TN += 1
            else:
                if true_value == 'true':
                    FP += 1
                    # print(i)
                else:
                    FN += 1
                    # print(i)
    
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FN + FP)    
        results[f"phase_{phase}"] = {"Recall": recall, "Precision": precision, "F1-score": f1, "Accuracy": accuracy}

    # Print results
    # print(f"Phase Evaluation:  Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    print(results)

    return results
 

# Function to filter and convert valid numeric values
# Evaluation function for numerical properties
def property_evaluate(LLM_data, manual_data, property_keys, tolerance):
    results = {}
    
    for prop in property_keys:
        TP, TN, FP, FN = 0, 0, 0, 0
        for t, p in zip(manual_data, LLM_data):
            tv = (float(t[prop]) if t[prop] != "N" else 0.0)
            pv = (float(p[prop]) if p[prop] != "N" else 0.0)

            if tv == 0 and pv == 0:
                # TN += 1
                continue
            elif tv == 0:
                FP += 1
            elif tv != 0:
                if abs(tv - pv)/tv < tolerance:
                    TP += 1
                else:
                    FN += 1
                    # print(prop, tv, pv)
        
        print(TP, TN, FP, FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FN + FP)

        results[f"Property_{prop}"] = {"Recall": recall, "Precision": precision, "F1-score": f1, "Accuracy": accuracy}

    print(results)

    return results


if __name__ == "__main__":
    # Load JSONL data
    LLM_data= read_jsonl('./data_evaluate/dataset-extract-by-V3+R1.jsonl')
    manual_data = read_jsonl('./data_evaluate/data_test-V3+R1.jsonl')
    composition_evaluate(LLM_data, manual_data, tolerance=0.5)
    phase_evaluate(LLM_data, manual_data)
    property_keys = ["density(g/cm³)", "hardness(HV)", "compressive_yield_strength_at_room_temperature(MPa)", "compressive_strain_at_room_temperature(%)", 
             "compressive_yield_strength_at_1073K(MPa)", "compressive_yield_strength_at_1273K(MPa)"]
    property_evaluate(LLM_data, manual_data, property_keys, tolerance = 0.05)

