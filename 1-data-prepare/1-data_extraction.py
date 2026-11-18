import os
import json
from typing import Dict
import re
import pandas as pd
from openai import OpenAI, Timeout
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered



# ========== 配置部分 ==========
CONFIG = {
    "base_url": "Your API base URL",###
    "deepseek_key": "Your API key",###
    "model_name_extraction": "DeepSeek-V3-P001",
    "model_name_checking": "DeepSeek-R1",
    "pdf_folder": "./src-pdf",###
    "output_folder": "./extraction-output",
    "failed_file": "./extraction-output/failed.txt",
    "processed_file": "./extraction-output/processed.txt",

    "query_template": """Please extract the alloy composition, preparation process, microstructure, and mechanical properties information reported in the literature on high-entropy alloys, and output the results in JSON format:
                         Output requirements:
                         - If there are multiple components or preparation processes in the paper, output multiple results in standard json format, and collect complete data as much as possible.
                         - Convert the nominal composition formulation to a uniform format, such as converting AlNb2TiVZr0.2 to Al19.23Nb28.46Ti19.23V19.23Zr3.85, converting (NbMoTiSi0.3)99.5La0.5 to Nb30.15Mo30.15Ti30.15Si9.05La0.5, 
                           make sure that the total sum 100.
                         - In process_method_1, determine which alloy preparation process is used, only choose in arc melting/additive manufacturing or powder metallurgy.
                         - Fill in process_method_2 with "method name+post process1-post process2.." like "vacuum arc melting + anneal-rolling", "spark plasma sintering+anneal","selective laser melting"
                         - Carefully distinguishing the preparation and processing methods for each alloy.The process_description is standardized to start with “prepared by...”. 
                           Each json form a continuous and complete description of the preparation process including specific process parameters in the content. Do not use abbreviations.
                         - Fill in phase according to "phase1+phase2+.." format. Determine if there are compounds or oxides in "intermetallic_phase".
                         - Note that the unit in parentheses, if the extracted information unit is inconsistent, a unit conversion is needed, such as hardness with MPa or GPa unit must be transfered with 4500MPa to 459HV.
                         - Extract as much information as possible especially data in tables. Attention to different expressions for the same property. The property data fill in specific value not a range, e.g. >40% take 40%. Non-existent data fill "N".
                         - Do not add other json fields not included in the template.
                         """,

    "json_template": """{ "doi": "10.1016/j.ijrmhm.2020.105357",
                            "title": "Microstructure and mechanical properties of TiNbV0.5Ta0.5Crx (x=0, 0.1, 0.2, 0.5) refractory high-entropy alloys",
                            "composition_formula":"Ti28.58Nb28.58V14.28Ta14.28Cr14.28",
                            "process_method_1":"arc melting/additive manufacturing/powder metallurgy",
                            "process_method_2":"vacuum arc melting+anneal-rolling",
                            "process_description":"prepared by..",
                            "phase":"BCC_A2/BCC+C14Laves/..",
                            "single_solution_phase":"True/False",
                            "intermetallic_phase":"True/False",
                            "density(g/cm³)":"6.8",
                            "hardness(HV)":"540",
                            "compressive_yield_strength_at_room_temperature(MPa)":"2000",
                            "compressive_strain_at_room_temperature(%)":"14",
                            "tensile_yield_strength_at_room_temperature(MPa)":"1200",
                            "tensile_strain_at_room_temperature(%)":"10",
                            "compressive_yield_strength_at_1073K(MPa)":"800",
                            "compressive_yield_strength_at_1273K(MPa)":"200"
                        }""",

    "json_checking_prompt": """ You are a professional json data checking expert.
                                - You are given a json data and a json schema. Your task is to check whether the json data is consistent with the json schema. Check that all required information in the template is included.
                                  Delete redundant json fields not included in the template.
                                - Double-check the process_description and each property index, if it contains more than one preparation process parameters or property data in the same index, 
                                please modify it according to the output format to output more than one json respectively, make sure each process_description and property index are complete, without unnecessary explanation.
                                - Check whether the alloy composition is wrong. Make ensure that strictly follow nominal compositions formula and the element is right and the atomic sum is 100. 
                                - if there are something wrong in format or data, then according to the output requirements to refill, do not need unnecessary explanation.”
                            """,
    "data_checking": "TRUE",
                                           
}

os.makedirs(CONFIG["output_folder"], exist_ok=True)

# ========== 辅助函数 ==========
def load_processed_records(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

def save_processed_record(file_path, filename):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

def save_failed_record(file_path, filename, error):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{filename}|{error}\n")

# ========== PDF 处理器 ==========
class PDFProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """文本清洗、过滤各类Introduction标题和参考文献"""
        # 处理连字符换行（保留原逻辑）
        text = re.sub(r'-\n(\w)', r'\1', text)
          
        # 精准匹配Introduction标题（井号开头）(去除易导致doi提取失败)
        intro_pattern = re.compile(r'^#+\s*[^\n]*?introduction[^\n]*', flags=re.IGNORECASE | re.MULTILINE)
        match = intro_pattern.search(text)
        if match:
            print("Introduction was found and deleted")
            # 查找下一个井号标题或文档结尾
            end_match = re.search(r'(^#+\s*[^\n]*|\Z)', text[match.end():], flags=re.MULTILINE)
            if end_match:
                cut_position = match.end() + end_match.start()
                
                # 提取Introduction部分的内容
                intro_content = text[match.start():cut_position]
                
                # 匹配doi所在行，支持灵活格式
                doi_pattern = re.compile(r'^.*10\.\d{4,9}/[-._;()/:A-Z0-9]+.*$', flags=re.IGNORECASE | re.MULTILINE)
                doi_match = doi_pattern.search(intro_content)
                
                if doi_match:
                    # 提取包含doi的完整行
                    doi_line_start = doi_match.start()
                    doi_line_end = doi_match.end()
                    
                    # 找到行的起始和结束位置
                    line_start = intro_content.rfind('\n', 0, doi_line_start) + 1
                    line_end = intro_content.find('\n', doi_line_end)
                    if line_end == -1:
                        line_end = len(intro_content)
                    
                    doi_line = intro_content[line_start:line_end].strip()
                    
                    # 截断Introduction部分，但保留doi所在行
                    text = text[:match.start()] + doi_line + "\n" + text[cut_position:]
                else:
                    # 如果没有找到doi，直接截断Introduction部分
                    text = text[:match.start()] + text[cut_position:]
        else:
            print("Introduction was not found")
        
        #去除参考文献
        for keyword in ['acknowledgment','references', 'bibliography', 'reference',"参考文献"]:
            pattern = re.compile(rf'^#+\s*[^\n]*?{re.escape(keyword)}[^\n]*', re.IGNORECASE | re.MULTILINE)
            match = pattern.search(text)
            if match:
                text = text[:match.start()]
                print("References was found and deleted")
                break
        print("Text cleaned with total length:", len(text), "characters")
        return text.strip()

    @staticmethod
    def read_pdf(file_path: str) -> str:
        # try:
            converter = PdfConverter(artifact_dict=create_model_dict(device="cuda"))
            rendered = converter(file_path)
            full_text, _, images = text_from_rendered(rendered)
            print("PDF to markdown completed.")
            return PDFProcessor.clean_text(full_text)
        # except Exception as e:
        #     raise RuntimeError(f"PDF读取失败: {str(e)}")

    @staticmethod
    def get_unprocessed_files():
        processed = load_processed_records(CONFIG["processed_file"])
        failed = load_processed_records(CONFIG["failed_file"])
        # 清空 failed 文件内容
        with open(CONFIG["failed_file"], 'w', encoding='utf-8') as f:
            f.write("")  # 清空文件内容
        all_files = [f for f in os.listdir(CONFIG["pdf_folder"]) if f.endswith('.pdf')]
        return [f for f in all_files if f not in processed]

# ========== DeepSeek 客户端 ==========
class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=CONFIG["deepseek_key"],
            base_url=CONFIG["base_url"],
        )
        self.query_template = CONFIG["query_template"]

        self.json_template = CONFIG["json_template"]

    def extract_knowledge(self, content: str) -> Dict:
        content = content[:70000]
        user_prompt = self.query_template + "\ncontent:" + content
        print("Data Extracting ...")

        try:
            # 1. json extracting
            messages = [
                {"role": "system", "content": f"You are a professional materials science assistant who needs to accurately parse technical documents with fellowing template {self.json_template}"},
                {"role": "user", "content": user_prompt},
            ]
            response = self.client.chat.completions.create(
                model= CONFIG["model_name_extraction"],
                messages= messages,
                temperature=1,
                stream=False,
                max_tokens= 8000,
                response_format={"type": "json_object"}
            )

            # json checking on
            if CONFIG["data_checking"] == "TRUE":
                messages.append(response.choices[0].message)
                result1 = response.choices[0].message.content 
                print(f"extracted json: {result1}")

                # 2.json checking
                print("json checking")
                messages.append({"role": "user", 
                                "content": f"{CONFIG['json_checking_prompt']}"})

                response = self.client.chat.completions.create(
                    model=CONFIG["model_name_checking"],#DeepSeek-R1
                    # model="DeepSeek-R1-N001",
                    messages= messages,
                    temperature=0.6,
                    stream=False,
                    max_tokens= 8000,
                    response_format={"type": "json_object"}
                )
                result = response.choices[0].message.content    
                print(f"checked json: {result}")
                json_matches = re.findall(r'```json\n(.*?)\n```', result, re.DOTALL)
            # print(f"转换结果: {json_matches}")

            else : #json checking off
                result = response.choices[0].message.content 
                print(f"extracted json: {result}")
                json_matches = re.findall(r'```json\n(.*?)\n```', result, re.DOTALL)

            
            # 3. json formation
            parsed_results = []
            try:
                for match in json_matches:
                    parsed_results.append(json.loads(match))
            
            except json.JSONDecodeError as e:
                # print(f"JSON解析错误: {str(e)}")
                for match in json_matches:
                    json_objects = re.split(r'(?<=\})\s*(,|\n)\s*(?=\{)', match)
                    # 去掉分隔符号
                    json_objects = [obj.strip() for obj in json_objects if obj.strip()]
                    for json_obj in json_objects:
                        parsed_results.append(json.loads(json_obj))

            return  parsed_results
        
        except:
            
            return []

# ========== 主流程 ==========
def main():
    processor = PDFProcessor()
    deepseek = DeepSeekClient()
    
    # 获取待处理文件列表
    files_to_process = processor.get_unprocessed_files()
    if not files_to_process:
        print("没有需要处理的新文件")
    file_num = len(files_to_process)
    # print(files_to_process)
    # print(file_num)
    # exit(0)

    output_dir = os.path.join(CONFIG["output_folder"], 'json-result/')
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    # 批量处理PDF
    for i, filename in enumerate(files_to_process, start=1):
        base_name = os.path.splitext(filename)[0][:150]  # 获取文件名（去掉扩展名）
        output_file_path = os.path.join(output_dir, f'{base_name}.jsonl')
  
        try: 
            print(f"{i}/{file_num} 正在处理文件: {filename}")
            text = processor.read_pdf(os.path.join(CONFIG["pdf_folder"], filename))
            future = deepseek.extract_knowledge(text)
            # print(future)
            # exit(0)
            
            if not future:# 判断future是否为空列表
                # 记录处理失败
                save_failed_record(CONFIG["failed_file"], filename, 'No results returned')
                print("JSON提取失败: 没有返回结果") 
                continue

            elif isinstance(future, list):
                flattened_future = [item for sublist in future for item in (sublist if isinstance(sublist, list) else [sublist])]

            else:
                flattened_future = [future]

            with open(output_file_path, 'a', encoding='utf-8') as f:
                for item in flattened_future:
                    if isinstance(item, dict):
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')                        
                    else:
                        # 记录处理失败
                        save_failed_record(CONFIG["failed_file"], filename, f'Unexpected item type: {type(item)}')
                        print(f"处理失败: 意外的项目类型 - {type(item)}")
                
            # 记录处理成功
            save_processed_record(CONFIG["processed_file"], filename)
            print(f"成功处理文件: {filename}")

        except Exception as e:
            # 记录处理失败
            save_failed_record(CONFIG["failed_file"], filename, str(e))
            print(f"Error: {str(e)}") 


            

if __name__ == "__main__":
    main()
