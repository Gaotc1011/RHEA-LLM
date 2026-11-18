import json
import pandas as pd

def convert_jsonl_to_excel(input_path, output_path):
    """
    Convert JSONL file to Excel format.
    """
    data_list = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
        
        df = pd.DataFrame(data_list)
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Conversion successful: {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    input_file = "alloy_database-all.jsonl"
    output_file = "alloy_database-all.xlsx"
    convert_jsonl_to_excel(input_file, output_file)