import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict

# 配置参数
file_path = './extraction-output/alloy_database-all.jsonl'  # 替换为你的JSONL文件路径
output_dir = "./data_analysis/plots"
stats_dir = "./data_analysis/stats"


# 目标性能指标
target_properties = [
    "density(g/cm³)",
    "hardness(HV)",
    "compressive_yield_strength_at_room_temperature(MPa)",
    "compressive_strain_at_room_temperature(%)",
    "compressive_yield_strength_at_1073K(MPa)",
    "compressive_yield_strength_at_1273K(MPa)"
]

# 读取JSONL文件并提取数据
data = {prop: [] for prop in target_properties}

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        for prop in target_properties:
            value = entry.get(prop, "N")
            if value != "N" and value != "":
                try:
                    data[prop].append(float(value))
                except (ValueError, TypeError):
                    pass

# 统计数据并生成报告
statistics_report = "性能指标统计报告\n\n"
stats_df= pd.DataFrame()


for prop, values in data.items():
    
    # 计算统计指标
    count = len(values)
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    p25 = np.percentile(values, 25)
    median = np.median(values)
    p75 = np.percentile(values, 75)

    # 添加到文本报告
    statistics_report += f"{prop}:\n"
    statistics_report += f"  样本数量: {count}\n"
    statistics_report += f"  平均值: {mean:.4f}\n"
    statistics_report += f"  标准差: {std:.4f}\n"
    statistics_report += f"  最小值: {min_val:.4f}\n"
    statistics_report += f"  25%分位数: {p25:.4f}\n"
    statistics_report += f"  中位数: {median:.4f}\n"
    statistics_report += f"  75%分位数: {p75:.4f}\n"
    statistics_report += f"  最大值: {max_val:.4f}\n\n"

    # 保存统计结果
    # convert to dataframe
    stat_data = pd.DataFrame(values, columns=[prop])
    stats_df = pd.concat([stats_df, stat_data], axis=1)
    
    # 绘制直方图
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 美化图表
    plt.title(f'{prop} distribution', fontsize=14)
    plt.xlabel(prop.split('(')[0] + f" ({prop.split('(')[1].split(')')[0]})", fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加统计信息框
    stats_text = (f"Count: {count}\n"
                  f"Average: {mean:.2f}\n"
                  f"Std: {std:.2f}\n"
                  f"Median: {median:.2f}")
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
    
    # 保存图像
    filename = prop.replace('(', '').replace(')', '').replace('/', '_')
    plt.savefig(os.path.join(output_dir, f"{filename}_histogram.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

# 保存统计报告
with open(os.path.join(stats_dir, "property_statistics_report.txt"), 'w', encoding='utf-8') as f:
    f.write(statistics_report)

# 保存为CSV文件
stats_df.to_csv(os.path.join(stats_dir, "property_statistics_summary.csv"), index=False)

print(f"处理完成！结果保存在 {output_dir} 目录中")
print(f"共处理 {len(target_properties)} 个性能指标")
print(f"生成的报告包含:")
print(f"  - 统计报告文本文件: statistics_report.txt")
print(f"  - CSV汇总表格: statistics_summary.csv")
print(f"  - {len(target_properties)} 个直方图PNG文件")





# phase analysis
# 读取JSONL文件并提取数据
phase_data = {"Single SS": 0, "SS+IM": 0, "Multi SS":0}

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        ss, im = entry["single_solution_phase"], entry["intermetallic_phase"]

        if ss == "False" and im == "True":
            phase_data["SS+IM"] += 1
        elif ss == "True" and im  == "False":
            phase_data["Single SS"] += 1
        elif ss  == "False" and im  == "False":
            phase_data["Multi SS"] += 1
print(phase_data)


# Prepare data for visualization
labels = list(phase_data.keys())
sizes = list(phase_data.values())
total = sum(sizes)
percentages = [100 * (size / total) for size in sizes]

# Custom colors
# colors = ['#66b3ff', '#99ff99', '#ffcc99']
colors = plt.cm.Pastel1(np.linspace(0, 1, len(sizes)))

# Create pie chart
plt.figure(figsize=(10, 8), dpi=100)

# Draw pie chart
wedges, texts, autotexts = plt.pie(
    sizes,
    # labels=labels,
    colors=colors,
    autopct=lambda p: f'{p:.1f}%' if p > 1 else '',  # Only show percentages > 1%
    startangle=90,
    pctdistance=0.85,
    textprops={'fontsize': 12},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)

# Add white circle in center (donut chart style)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)

# Equal aspect ratio ensures pie is drawn as a circle
plt.axis('equal')

# Add title and statistics
plt.title('Phase Distribution in Alloys', pad=20, fontsize=14, weight='bold')

# Create detailed legend
legend_labels = [f"{label}: {size} alloys ({pct:.1f}%)" 
                for label, size, pct in zip(labels, sizes, percentages)]
plt.legend(wedges, legend_labels,
    title="Phase Types",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)

# # Add total count annotation
# plt.text(0, 0, f"Total: {total}", 
#          horizontalalignment='center',
#          verticalalignment='center',
#          fontsize=12)

# Save plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'phase_distribution_pie.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# Print phase statistics
print("\nPhase Statistics:")
for label, size, pct in zip(labels, sizes, percentages):
    print(f"{label}: {size} alloys ({pct:.1f}%)")
print(f"Total alloys analyzed: {total}")
print(f"Phase distribution plot saved to: {os.path.join(output_dir, 'phase_distribution_pie.png')}")




# --- Process Method Analysis ---
# Count occurrences of each process_method_1 value
process_method_counter = defaultdict(int)

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        method = entry.get("process_method_1", "Unknown")  # Treat missing as "Unknown"
        process_method_counter[method] += 1

# Prepare data for visualization
labels = list(process_method_counter.keys())
sizes = list(process_method_counter.values())
total = sum(sizes)
percentages = [100 * (size / total) for size in sizes]

# Filter out small slices for cleaner display
MIN_PERCENTAGE = 1.0
filtered_labels = [label for label, pct in zip(labels, percentages) if pct >= MIN_PERCENTAGE]
filtered_sizes = [size for size, pct in zip(sizes, percentages) if pct >= MIN_PERCENTAGE]
other_count = sum(size for size, pct in zip(sizes, percentages) if pct < MIN_PERCENTAGE)

if other_count > 0:
    filtered_labels.append("Other (<1%)")
    filtered_sizes.append(other_count)

# Recalculate percentages after filtering
filtered_total = sum(filtered_sizes)
filtered_percentages = [100 * (size / filtered_total) for size in filtered_sizes]

# Custom colors
colors = plt.cm.Pastel1(np.linspace(0, 1, len(filtered_sizes)))

# Create pie chart
plt.figure(figsize=(12, 8), dpi=300)

wedges, texts, autotexts = plt.pie(
    filtered_sizes,
    # labels=filtered_labels,
    colors=colors,
    autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
    startangle=90,
    pctdistance=0.85,
    textprops={'fontsize': 12},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)

# Add white circle in center (donut chart style)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)

# Equal aspect ratio ensures pie is drawn as a circle
plt.axis('equal')

# Add title
plt.title('Distribution of Process Method 1', pad=20, fontsize=14, weight='bold')
legend_labels = [f"{label}: {size} alloys ({pct:.1f}%)" 
                for label, size, pct in zip(filtered_labels, filtered_sizes, filtered_percentages)]
plt.legend(wedges, legend_labels,
    title="Phase Types",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)

# Save plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'process_method_distribution_pie.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# Print statistics
print("\nProcess Method 1 Statistics:")
for label, size, pct in zip(filtered_labels, filtered_sizes, filtered_percentages):
    print(f"{label}: {size} entries ({pct:.1f}%)")
print(f"Total entries analyzed: {total}")
print(f"Process method distribution plot saved to: {os.path.join(output_dir, 'process_method_distribution_pie.png')}")