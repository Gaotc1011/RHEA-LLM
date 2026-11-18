import json
import re
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import pandas as pd
import os
from collections import defaultdict
import csv



# 设置绘图风格
# sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14
})

# 1. 读取并解析JSONL文件
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

# 2. 提取元素及其含量
def extract_elements(compositions):
    element_counter = defaultdict(int)
    element_concentrations = defaultdict(list)
    pattern = r'([A-Z][a-z]?)(\d+\.?\d*|\d*\.\d+)'  # 匹配元素符号和含量
    
    for comp in compositions:
        matches = re.findall(pattern, comp)
        for element, conc in matches:
            # remove conc=0
            try:
                if float(conc) == 0:
                    continue
                concentration = float(conc)
                element_counter[element] += 1
                element_concentrations[element].append(concentration)
            except ValueError:
                continue
    
    return element_counter, element_concentrations

# 3. 统计每个合金的元素个数
def count_elements_per_alloy(compositions):
    element_counts = []
    pattern = r'([A-Z][a-z]?)\d+\.?\d*'  # 匹配元素符号
    
    for comp in compositions:
        elements = re.findall(pattern, comp)
        element_counts.append(len(set(elements)))  # 使用set确保唯一元素
    
    return element_counts

# 4. 绘制元素频率条形图
def plot_element_frequency(element_counter, compositions, output_dir="plots"):
    if not element_counter:
        print("No valid data to plot")
        return
    
    # 按元素频率排序
    sorted_elements = sorted(element_counter.items(), key=lambda x: x[1], reverse=True)
    element_names = [e[0] for e in sorted_elements]
    element_counts = [e[1] for e in sorted_elements]
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制条形图
    bars = plt.bar(element_names, element_counts, color='steelblue')
    plt.title('Frequency of Elements in Alloys', pad=20)
    plt.ylabel('Number of Alloys')
    plt.xlabel('Chemical Elements')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'element_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved element frequency plot to 'plots/element_frequency.png'")

# 5. 绘制带数据点的元素含量箱线图
def plot_concentration_boxplot(element_counter, element_concentrations, output_dir="plots"):
    if not element_concentrations:
        print("No valid data to plot")
        return
    
    # 按元素频率排序
    sorted_elements = sorted(element_counter.items(), key=lambda x: x[1], reverse=True)
    element_names = [e[0] for e in sorted_elements]
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 准备箱线图数据
    data = [element_concentrations[e] for e in element_names]
    
    # 绘制箱线图
    box = plt.boxplot(data, patch_artist=True, showfliers=False)
    
    # 设置箱线图颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(element_names)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 添加所有数据点（带抖动）
    for i, concentrations in enumerate(data):
        # 添加抖动避免点重叠
        x = np.random.normal(i + 1, 0.04, len(concentrations))
        plt.scatter(x, concentrations, color='darkslategray', alpha=0.5, s=30, edgecolor='w')
    
    plt.title('Distribution of Element Concentrations', pad=15)
    plt.ylabel('Concentration (at.%)')
    plt.xlabel('Chemical Elements')
    plt.xticks(range(1, len(element_names) + 1), element_names, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'element_concentrations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved concentration distribution plot to 'plots/element_concentrations.png'")

# 6. 绘制元素个数分布的饼图
def plot_element_count_pie(element_counts, output_dir="plots"):
    if not element_counts:
        print("No element count data to plot")
        return
    
    # 统计不同元素个数的合金数量
    count_freq = {}
    for count in element_counts:
        count_freq[count] = count_freq.get(count, 0) + 1
    
    # 按元素个数排序
    sorted_counts = sorted(count_freq.items(), key=lambda x: x[0])
    counts = [f"{c} elements" for c, _ in sorted_counts]
    freqs = [f for _, f in sorted_counts]
    percentages = [f/sum(freqs)*100 for f in freqs]
    
    # 创建饼图
    plt.figure(figsize=(10, 8))
    
    # 设置颜色
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(counts)))
    
    # 绘制饼图
    wedges, texts, autotexts = plt.pie(
        freqs, 
        # labels=counts, 
        colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',  # 只显示大于3%的百分比
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 12}
    )
    
    # 设置饼图中间的空白
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # 添加标题
    plt.title('Distribution of Element Counts per Alloy', pad=20)
    
    # 添加图例
    legend_labels = [f"{c}: {f} alloys ({p:.1f}%)" for c, f, p in zip(counts, freqs, percentages)]
    plt.legend(wedges, legend_labels, 
               title="Element Counts",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))
       
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'element_count_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved element count pie chart to 'plots/element_count_pie.png'")

# Add this function to create a chord diagram
def plot_element_cooccurrence_chord_hv(compositions, selected_elements=None, output_dir="plots"):
    """
    Plot a chord diagram showing co-occurrence of elements in alloy compositions using holoviews.
    
    Parameters:
    compositions (list): List of composition formulas
    selected_elements (list): List of specific elements to analyze (if None, use all elements)
    output_dir (str): Directory to save the plot
    """

    # Extract element pairs from compositions
    element_pairs = defaultdict(int)
    element_counter = defaultdict(int)
    pattern = r'([A-Z][a-z]?)\d*\.?\d*'  # Match element symbols
    
    for comp in compositions:
        elements = re.findall(pattern, comp)
        # Count individual elements
        for element in elements:
            element_counter[element] += 1
        # Count element pairs (undirected)
        unique_elements = list(set(elements))
        for i in range(len(unique_elements)):
            for j in range(i+1, len(unique_elements)):
                # Sort to ensure consistent ordering
                pair = tuple(sorted([unique_elements[i], unique_elements[j]]))
                element_pairs[pair] += 1
    
    # Filter elements if specific list provided
    if selected_elements:
        # Filter pairs to only include selected elements
        filtered_pairs = {}
        for pair, count in element_pairs.items():
            if pair[0] in selected_elements and pair[1] in selected_elements:
                filtered_pairs[pair] = count
        element_pairs = filtered_pairs
        
        # Filter element counter
        filtered_counter = defaultdict(int)
        for element in selected_elements:
            filtered_counter[element] = element_counter.get(element, 0)
        element_counter = filtered_counter
        elements_list = selected_elements
    else:
        # Use all elements if none specified
        elements_list = sorted(element_counter.keys())
    
    # Create a DataFrame for the chord diagram
    if len(elements_list) < 2:
        print("Need at least 2 elements to create chord diagram")
        return
    
    # Create matrix of co-occurrences
    n_elements = len(elements_list)
    element_to_index = {elem: i for i, elem in enumerate(elements_list)}
    
    # Initialize matrix
    matrix = np.zeros((n_elements, n_elements))
    
    # Fill matrix with co-occurrence data
    for (elem1, elem2), count in element_pairs.items():
        if elem1 in element_to_index and elem2 in element_to_index:
            i, j = element_to_index[elem1], element_to_index[elem2]
            matrix[i, j] = count
            matrix[j, i] = count  # Symmetric matrix

    # Create chord diagram using holoviews from links
    df = pd.DataFrame(matrix.astype(int), index=elements_list, columns=elements_list)
    df.to_csv(f"./stats/element_cooccurrence.csv")
    print(df)

    # matrix_list = matrix.astype(int).tolist()
    # elements_list_copy = elements_list[:]  # Ensure it's a regular list

    # from chord import Chord
    # chord_diagram = Chord(matrix_list, elements_list_copy, colors="d3.schemeSet2").to_html()
    # output_file = os.path.join(output_dir, 'element_cooccurrence_chord.html')
    # # chord_diagram.to_html(output_file)  # Save to file
    # print(f"Saved chord diagram to {output_file}")

# 7. 保存统计结果为CSV文件
def save_statistics(element_counter, element_concentrations, element_counts, alloy_data, output_dir="stats"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 元素频率统计
    element_freq_file = os.path.join(output_dir, "element_frequency.csv")
    with open(element_freq_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Element", "Frequency", "Percentage", "Min Concentration", 
                         "Max Concentration", "Median Concentration", "Mean Concentration"])
        
        total_alloys = len(element_counts)
        for element, count in sorted(element_counter.items(), key=lambda x: x[1], reverse=True):
            concs = element_concentrations[element]
            writer.writerow([
                element,
                count,
                f"{count/total_alloys*100:.2f}%",
                min(concs),
                max(concs),
                np.median(concs),
                np.mean(concs)
            ])
    print(f"Saved element frequency stats to {element_freq_file}")
    
    # 2. 元素个数分布统计
    count_dist_file = os.path.join(output_dir, "element_count_distribution.csv")
    count_freq = {}
    for count in element_counts:
        count_freq[count] = count_freq.get(count, 0) + 1
    
    with open(count_dist_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Element Count", "Frequency", "Percentage"])
        
        total = sum(count_freq.values())
        for count in sorted(count_freq.keys()):
            freq = count_freq[count]
            writer.writerow([
                count,
                freq,
                f"{freq/total*100:.2f}%"
            ])
    print(f"Saved element count distribution to {count_dist_file}")
    
    # 3. 每个合金的元素个数
    alloy_count_file = os.path.join(output_dir, "alloy_element_counts.csv")
    with open(alloy_count_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["DOI", "Title", "Composition", "Element Count"])
        
        for i, data in enumerate(alloy_data):
            comp = data.get('composition_formula', '')
            title = data.get('title', '')
            doi = data.get('doi', '')
            count = element_counts[i] if i < len(element_counts) else 0
            
            writer.writerow([
                doi,
                title[:100] + "..." if len(title) > 100 else title,  # 截断长标题
                comp,
                count
            ])
    print(f"Saved per-alloy element counts to {alloy_count_file}")
    
    # 4. 元素浓度详细数据
    conc_detail_file = os.path.join(output_dir, "element_concentration_details.csv")
    with open(conc_detail_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Element", "Concentration"])
        
        for element, concs in element_concentrations.items():
            for conc in concs:
                writer.writerow([element, conc])
    print(f"Saved element concentration details to {conc_detail_file}")

# 主函数
def main():
    file_path = './extraction-output/alloy_database-all.jsonl'  # 替换为实际文件路径
    output_dir = "./data_analysis/plots"
    stats_dir = "./data_analysis/stats"
    
    # 处理数据
    compositions, alloy_data = parse_jsonl(file_path)
    if not compositions:
        print("No composition data found in the file.")
        return
    
    element_counter, element_concentrations = extract_elements(compositions)
    element_counts = count_elements_per_alloy(compositions)

    # 打印元素统计结果
    total_alloys = len(compositions)
    print(f"Total alloys processed: {total_alloys}")
    print(f"Unique elements found: {len(element_counter)}")
    print(f"Average elements per alloy: {sum(element_counts)/total_alloys:.2f}")
    
    # 打印元素频率
    print("\nTop 10 elements by frequency:")
    for element, count in sorted(element_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
        concs = element_concentrations[element]
        print(f"{element}: {count} alloys ({count/total_alloys*100:.1f}%) | "
              f"Conc: {min(concs):.1f}-{max(concs):.1f}% (median: {np.median(concs):.1f}%)")
    
    # 打印元素个数分布
    count_freq = {}
    for count in element_counts:
        count_freq[count] = count_freq.get(count, 0) + 1
    
    print("\nElement count distribution:")
    for count in sorted(count_freq.keys()):
        freq = count_freq[count]
        print(f"{count} elements: {freq} alloys ({freq/total_alloys*100:.1f}%)")
    
    # 绘制图表
    plot_element_frequency(element_counter, compositions, output_dir)
    plot_concentration_boxplot(element_counter, element_concentrations, output_dir)
    plot_element_count_pie(element_counts, output_dir)
    plot_element_cooccurrence_chord_hv(compositions, 
                                      selected_elements=["Nb","Ti","V","Zr","Ta","Mo","Al","Hf","Cr","W"], 
                                      output_dir=output_dir)
    
    
    # 保存统计结果
    save_statistics(element_counter, element_concentrations, element_counts, alloy_data, stats_dir)
    
    print("\nAll plots saved to 'plots' directory")
    print("All statistics saved to 'stats' directory")

if __name__ == '__main__':
    main()