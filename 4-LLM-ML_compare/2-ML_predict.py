import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)



def parse_composition(comp_str):
    """
    Parse composition string to dictionary
    
    Args:
        comp_str (str): Composition in format Nb25Ti25V25Al25
        
    Returns:
        dict: Element compositions
    """
    composition = {element: 0 for element in elements}
    while comp_str:
        for element in elements:
            if comp_str.startswith(element):
                # Find number after element symbol
                num_str = ""
                for char in comp_str[len(element):]:
                    if char.isdigit() or char == '.':
                        num_str += char
                    else:
                        break
                composition[element] = float(num_str) if num_str else 1.0
                comp_str = comp_str[len(element)+len(num_str):]
                break
    return composition

def predict_multiple_properties(compositions_list):
    """
    Predict multiple properties for multiple alloy compositions
    
    Args:
        compositions_list (list): List of composition strings
        
    Returns:
        pd.DataFrame: Prediction results
    """
    results = []
    columns = ["Composition"] + tasks
    
    for comp_str in compositions_list:
        row = [comp_str]
        composition = parse_composition(comp_str)
        
        for task_name in tasks:
            model_path = f"./results/{task_name}_best.pth"
            input_size = len(elements)
            model = RegressionModel(input_size)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            input_data = torch.tensor([composition[element] for element in elements])
            
            with torch.no_grad():
                input_data = input_data.unsqueeze(0)
                prediction = model(input_data).item()
                row.append(prediction)
        
        results.append(row)
    
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("predictions.csv", index=False)
    return df

# Example usage:
compositions = ["Nb18.1V22.9Zr20.7Mo17.3Hf8.9W12.1","Nb21.9Ta15.1Mo17.6Hf15.8Cr11.0W18.6", "Ti6.6V29.3Mo19.0Hf13.8Cr12.0W19.3",
    "Nb18.0Ti22.1Zr33.5Mo5.3Cr9.7W11.4","Nb19.7Ti24.8Ta10.1Mo16.1Al18.5Cr10.8",
    "Ti24.7Zr18.1Mo14.8Al14.4Hf11.0Cr17.0","Nb26.9V33.5Zr8.1Mo16.9Al14.6"
]
elements = ["Al","Cr","Hf","Mo","Nb","Si","Ta","Ti","V","W","Zr"]
# tasks = ["compressive_yield_strength_at_room_temperature"]
tasks = ["compressive_yield_strength_at_room_temperature", "compressive_strain_at_room_temperature",
        "compressive_yield_strength_at_1073K",
        "compressive_yield_strength_at_1273K", "hardness", "density" ]
df_results = predict_multiple_properties(compositions)
print(df_results)