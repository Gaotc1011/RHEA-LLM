import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def data_reader(data_path):
    data = []
    exclude_fields = {'composition_formula', 'composition'}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            for key in list(item.keys()):  
                if key in exclude_fields:
                    continue
                if isinstance(item[key], str):
                    item[key] = float(item[key])
                else:
                    continue

            data.append(item)
    
    return data


def making_cluster(all_generations):
    results = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for data in all_generations:
        if "Al" in data["composition"] and "W" in data["composition"]:
            continue
        # data = json.loads(data)
        if data['density'] > 9 and data['strain'] > 12 and data['hardness'] >500:
            cluster1.append(data)
        elif data['density'] < 9 and data['density'] > 7.5 and data['strain'] > 12 and data['hardness'] >500:
            cluster2.append(data)
        elif data['density'] < 7.5 and data['strain'] > 12 and data['hardness'] >500:
            cluster3.append(data)
        else:
            continue

    results.append(cluster1)
    results.append(cluster2)
    results.append(cluster3)
    # print(len(cluster1), len(cluster2), len(cluster3))

    for  i, items in enumerate(results):
        jsonltocsv(items, f"./result/cluster{i}.csv")

        with open(f"./result/cluster{i}.jsonl", 'w', encoding="utf-8") as f:
            for item in items:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    return results

def making_cluster_exp(all_generations):
    results = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for data in all_generations:
        # data = json.loads(data)
        if float(data["density(g/cm³)"]) > 9:
            cluster1.append(data)
        elif float(data["density(g/cm³)"]) < 9 and float(data["density(g/cm³)"]) > 7.5:
            cluster2.append(data)
        elif float(data["density(g/cm³)"]) < 7.5:
            cluster3.append(data)
        else:
            continue

    results.append(cluster1)
    results.append(cluster2)
    results.append(cluster3)
    # print(len(cluster1), len(cluster2), len(cluster3))    
    return results     
    

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost (higher for maximization)
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # For minimization
            is_efficient[i] = True  # And keep self
    return is_efficient


def find_pareto_front(data, target, weights, pattern = "cluster1"):
    opposite_signs = [-1 if w > 0 else 1 for w in weights]

    # Extract objectives (minimize density, maximize strengths)
    objectives = []
    for alloy in data:
        obj = [alloy[t] * w for t, w in zip(target, opposite_signs)]
        objectives.append(obj)
    objectives = np.array(objectives).reshape(-1, len(target))

    pareto_mask = is_pareto_efficient(objectives)
    pareto_front = [alloy for alloy, is_pareto in zip(data, pareto_mask) if is_pareto]

    save_path = "./result/pareto_front"
    for i in range(len(target)):
        save_path+= f"_{target[i]}"
    
    jsonltocsv(pareto_front, save_path+ f"_{pattern}.csv")
    with open(save_path+ f"_{pattern}.jsonl", 'w') as f:
        for alloy in pareto_front:
            f.write(json.dumps(alloy) + '\n')

    return pareto_front
    

def plot_3D_scatter(data, experiment=None):
    df = pd.DataFrame(data)
    if experiment is not None:
        experiment_df = pd.DataFrame(experiment)
    
        
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all data points with improved color scheme
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], 
               c="#BC75C3", alpha=0.3, s=25, label='All Predicted Alloys', 
               edgecolors='none')
    
    # Plot experimental alloys with contrasting color
    # if experiment is not None:
    #     ax.scatter(experiment_df.iloc[:, 0],  # First column
    #                experiment_df.iloc[:, 1],  # Second column
    #                experiment_df.iloc[:, 2],  # Third column
    #                c='#e377c2', alpha=0.8, s=60, label='Experiment Alloys',
    #                edgecolors='black', linewidth=0.5)
    ["#d16ba5", "#c777b9", "#ba83ca", "#aa8fd8", "#9a9ae1", "#8aa7ec", "#79b3f4", "#69bff8", "#52cffe", "#41dfff", "#46eefa", "#5ffbf1"]
    ["#c6458e", "#ba489d", "#a94dac", "#9353b9", "#775ac4", "#606ad2", "#4178db", "#0085e2", "#009ce7", "#00afdf", "#00bfd0", "#29cdbe"]
    # Customize plot with better fonts and labels
    ax.set_xlabel(df.columns[0], fontsize=14, labelpad=15)
    ax.set_ylabel(df.columns[1], fontsize=14, labelpad=15)
    ax.set_zlabel(df.columns[2], fontsize=14, labelpad=15)
    
    # Invert x and y axes
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Improve tick parameters
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add legend with better positioning
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(f'result/{df.columns[0]}-{df.columns[1]}-{df.columns[2]}_3D_scatter.jpg', dpi=500)
    plt.show()

def plot_3D_scatter_cluster(data, pattern = "cluster1"):
    df = pd.DataFrame(data)
      
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all data points (gray, semi-transparent)
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], 
               c='gray', alpha=0.2, s=20)
     
    # Customize plot
    ax.set_xlabel(df.columns[0], fontsize=12, labelpad=10)
    ax.set_ylabel(df.columns[1], fontsize=12, labelpad=10)
    ax.set_zlabel(df.columns[2], fontsize=12, labelpad=10)
    # ax.set_title(f'3D Pareto Front for Iteration {iteration_num}', fontsize=14, pad=20)
    # Hide the axes
    # ax.axis('off')
    ax.set_xlim(200, 2000)
    ax.set_ylim(0, 1400)
    ax.set_zlim(5, 11)
    # Adjust legend and view angle
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(f'result/{df.columns[0]}-{df.columns[1]}-{df.columns[2]}_3D_scatter_{pattern}.jpg', dpi=500)
    plt.show()


def plot_2D_pareto(data, pareto_front, experiment=None, pattern = "cluster1"):
    df = pd.DataFrame(data)
    pareto_df = pd.DataFrame(pareto_front)
    if experiment is not None:
        experiment_df = pd.DataFrame(experiment)

    plt.figure(figsize=(8, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c='gray', alpha=0.2, s=20, label='All Predicted Alloys')
    if experiment is not None:
        plt.scatter(experiment_df.iloc[:, 0], experiment_df.iloc[:, 1], c='pink', alpha=1, s=50, label='Experiment Alloys')
    plt.scatter(pareto_df.iloc[:, 0], pareto_df.iloc[:, 1], c='b', s=80, marker='^', label='Pareto Front Alloys')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.savefig(f"result/{df.columns[0]}_{df.columns[1]}_pareto_front_{pattern}.jpg", dpi = 500)
    plt.show()

def data_process(data, target):
    extracted_data = []
    for item in data:
        filtered_item = {key: item[key] for key in target if key in item}
        extracted_data.append(filtered_item)
    return extracted_data

def jsonltocsv(data, outpath):
    df = pd.DataFrame(data)
    df.to_csv(outpath, index=False)


if __name__ == "__main__":

    data = data_reader("all_materials.jsonl")
    data_1 = data_process(data, target = ["YS-RT", "YS-1073","density"])
    jsonltocsv(data_1, "./result/all_materials_pre.csv")
    data_2 = data_process(data, target = [ "strain",  "YS-1273", "hardness"])
    
    experiment_data_1 = data_reader("./experiment_data/experiment_data_VAM.jsonl")
    experiment_1 = data_process(experiment_data_1, target = ["compressive_yield_strength_at_room_temperature(MPa)", "compressive_yield_strength_at_1073K(MPa)", "density(g/cm³)"])
    # experiment_target = data_process(experiment_data_1, target = ["compressive_yield_strength_at_room_temperature(MPa)", "compressive_strain_at_room_temperature(%)"])

    experiment_data_2 = data_reader("./experiment_data/experiment_data_VAM_4.jsonl")
    experiment_2 = data_process(experiment_data_2, target = ["compressive_strain_at_room_temperature(%)", "compressive_yield_strength_at_1273K(MPa)", "hardness(HV)", ])

    plot_3D_scatter(data_1, experiment_1)
    plot_3D_scatter(data_2, experiment_2)

    cluster_pre = making_cluster(data)
    cluster_exp = making_cluster_exp(experiment_data_1)
    # exit(0)

    for i in range(len(cluster_pre)):
        data_pre = data_process(cluster_pre[i], target = ["YS-RT", "YS-1073", "density"])
        plot_3D_scatter_cluster(data_pre, pattern = f"cluster{i}")

        pareto_data = find_pareto_front(data = cluster_pre[i], target = ["YS-RT", "YS-1073"], weights = [1,1], pattern = f"cluster{i}")
        pareto_data = data_process(data = pareto_data, target = ["YS-RT", "YS-1073"])

        predict_data = data_process(data = cluster_pre[i], target = ["YS-RT", "YS-1073"])
        data_exp = data_process(cluster_exp[i], target = ["compressive_yield_strength_at_room_temperature(MPa)", "compressive_yield_strength_at_1073K(MPa)"])
        plot_2D_pareto(predict_data, pareto_data, experiment = data_exp, pattern = f"cluster{i}")
