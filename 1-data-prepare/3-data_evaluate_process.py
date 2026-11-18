import json
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate 


# 1. 加载模型（本地路径或默认下载）
model = SentenceTransformer('./model/all-mpnet-base-v2')  # 可替换为本地模型路径


def tokenize_phrases(text):
    phrases = [p.strip().lower() for p in re.split(r'\+|,|;|\band\b|\bwith\b', text)]
    phrases = [re.sub(r'\s+', ' ', p) for p in phrases if p.strip()]
    return phrases
def normalize_temperature(text):
    return re.sub(r'(\d+)\s*[℃°]?C', r'\1°C', text)
def extract_ordered_llm_ngrams(text, ngram_range=(2, 5)):
    """
    分两步：
    1. 基于 + , ; and with 等结构性符号切分为短语；
    2. 在每个短语内按原顺序生成滑动窗口短句（3~5个词）。
    保证顺序、上下文、粒度可控。
    """
    # 步骤1：结构性分句
    phrases = [p.strip().lower() for p in re.split(r'\+|,|;|\band\b|\bwith\b', text)]
    phrases = [re.sub(r'\s+', ' ', p) for p in phrases if p.strip()]
    
    # 步骤2：每个短语滑窗提取
    result = []
    for phrase in phrases:
        tokens = phrase.split()
        for n in range(ngram_range[0], ngram_range[1]+1):
            for i in range(len(tokens) - n + 1):
                chunk = ' '.join(tokens[i:i+n])
                result.append(chunk) 
    return phrases + result

def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def compute_metrics(y_true, y_pred):
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))

# 4. 对每对描述进行匹配评估
# 输出评估指标
def compute_metrics(y_true, y_pred):
    print("📊 语义匹配评估指标：")
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(y_true, y_pred))

# 加载数据
manual_data = load_jsonl('./data_evaluate/data_test-V3+R1.jsonl')
llm_data = load_jsonl('./data_evaluate/dataset-extract-by-V3+R1.jsonl')
output_file = "./data_evaluate/process_evaluate_results_V3+R1.txt"


all_y_true = []
all_y_pred = []

threshold = 0.6 # 匹配阈值，可调

with open(output_file, 'w', encoding='utf-8') as f:
    # 遍历每一组数据
    for idx, (m, l) in enumerate(zip(manual_data, llm_data)):
        print(f"\n==== 🔎 对比样本 {idx + 1} ====")
        manual_phrases = tokenize_phrases(normalize_temperature(m["process_description"]))
        llm_phrases = extract_ordered_llm_ngrams(l["process_description"])

        if not manual_phrases or not llm_phrases:
            continue

        # 向量编码
        manual_embeddings = model.encode(manual_phrases, convert_to_tensor=True)
        llm_embeddings = model.encode(llm_phrases, convert_to_tensor=True)

        table_data = []

        for i, emb in enumerate(manual_embeddings):
            sims = util.cos_sim(emb, llm_embeddings)
            max_sim_idx = sims.argmax().item()
            max_sim = sims[0][max_sim_idx].item()
            matched_phrase = llm_phrases[max_sim_idx]

            matched = "✅" if max_sim >= threshold else "❌"
            all_y_true.append(1)
            all_y_pred.append(1 if max_sim >= threshold else 0)

            table_data.append([
                manual_phrases[i],
                matched_phrase,
                f"{max_sim:.3f}",
                matched
            ])

                # Generate table string
        table_str = tabulate(table_data, headers=["Manual Phrase", "Matched LLM Phrase", "Similarity", "Matched"], tablefmt="fancy_grid")
        # print(table_str)
        
        # Write only the table string to the file
        f.write(f"==== Evaluation Sample {idx + 1} ====\n")
        f.write(table_str)
        f.write("\n\n")


# 输出总评估结果
compute_metrics(all_y_true, all_y_pred)