import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,confusion_matrix

data_list = ["density(g/cm³)", "hardness(HV)", "compressive_yield_strength_at_room_temperature(MPa)", "compressive_strain_at_room_temperature(%)", 
             "compressive_yield_strength_at_1073K(MPa)", "compressive_yield_strength_at_1273K(MPa)","single_solution_phase", "intermetallic_phase"]

data_train_path = './multitask_results/train_predictions_Qwen3-8B-finetuned-full-all-0717-644.jsonl'
data_val_path = './multitask_results/val_predictions_Qwen3-8B-finetuned-full-all-0717-644.jsonl'
output_fold = './multitask_results'
TRAINING_METHOD = "full"  # 与训练时一致


def data_reader(data_list, data_path):
    data_predict = {field: [] for field in data_list}
    data_label = {field: [] for field in data_list}

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            predict = data['predict']
            predict_json = json.loads(predict)
            # print(predict_json)

            label = data['output']
            label_json = json.loads(label)
            # print(label_json)

            # for item in p
            for item in label_json.keys():

                data_predict[item].append(predict_json[item ]) 
                data_label[item].append(label_json[item])
                    

    return data_predict, data_label

def phase_evaluate(data_predict, data_label, item):
    fields = ["TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1-score"]
    results = {field: 0 for field in fields}
    # Count TP, TN, FP, FN
    for (predict_value, label_value) in zip(data_predict[item], data_label[item]):
        if predict_value == label_value and label_value == "True":
            results["TP"] += 1
        elif predict_value == label_value and label_value == "False":
            results["TN"] += 1
        elif predict_value != label_value and label_value == "True":
            results["FP"] += 1
        elif predict_value != label_value and label_value == "False":
            results["FN"] += 1

    # Calculate metrics and update results
    if results["TP"] + results["FP"] + results["FN"] > 0:
        results["Accuracy"] = (results["TP"] + results["TN"]) / (results["TP"] + results["TN"] + results["FP"] + results["FN"])
        results["Precision"] = results["TP"] / (results["TP"] + results["FP"])
        results["Recall"] = results["TP"] / (results["TP"] + results["FN"])
        results["F1-score"] = 2 * (results["Precision"] * results["Recall"]) / (results["Precision"] + results["Recall"])

    return results


def property_evaluate(data_predict, data_label, item):

    fields = ["r2", "mae", "mse", "rmse"]
    results = {field: 0 for field in fields}

    prediction = [float(value) for value in data_predict[item]]
    label = [float(value) for value in data_label[item]]
    # print(len(label))

    results["r2"] = r2_score(label, prediction)
    results["mae"] = mean_absolute_error(label, prediction)
    results["mse"] = mean_squared_error(label, prediction)
    results["rmse"] = np.sqrt(mean_squared_error(label, prediction))

    return results



data_pred_train, data_true_train = data_reader(data_list, data_train_path)
data_pred_val, data_true_val = data_reader(data_list, data_val_path)


for i, item in enumerate(data_list):
    if "phase" in item:#相分类任务
        phase_result_train = phase_evaluate(data_pred_train, data_true_train, item)
        phase_result_val = phase_evaluate(data_pred_val, data_true_val, item)

        print(f"{item}_train Accuracy: {phase_result_train['Accuracy']} Precision: {phase_result_train['Precision']} Recall: {phase_result_train['Recall']} F1-score: {phase_result_train['F1-score']}")
        print(f"{item}_val Accuracy: {phase_result_val['Accuracy']} Precision: {phase_result_val['Precision']} Recall: {phase_result_val['Recall']} F1-score: {phase_result_val['F1-score']}")

        plt.figure(figsize=(8, 6))
        # 绘制混淆矩阵图
        plt.imshow(np.array([[phase_result_train["TP"], phase_result_train["FP"]], [phase_result_train["FN"], phase_result_train["TN"]]]), cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.xticks([0, 1], ['True', 'False'])
        plt.yticks([0, 1], ['True', 'False'])
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'{item} Confusion Matrix Trainset')
        plt.text(0, 0, f'TP = {phase_result_train["TP"]}', ha='center', va='bottom')
        plt.text(1, 0, f'FP = {phase_result_train["FP"]}', ha='center', va='bottom')
        plt.text(0, 1, f'FN = {phase_result_train["FN"]}', ha='center', va='bottom')
        plt.text(1, 1, f'TN = {phase_result_train["TN"]}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_fold, f'{TRAINING_METHOD}_predictions_phase_train-0717-644-{i}.jpg'), dpi = 500)

        plt.figure(figsize=(8, 6))
        # 绘制混淆矩阵图
        plt.imshow(np.array([[phase_result_val["TP"], phase_result_val["FP"]], [phase_result_val["FN"], phase_result_val["TN"]]]), cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.xticks([0, 1], ['True', 'False'])
        plt.yticks([0, 1], ['True', 'False'])
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'{item} Confusion Matrix valset')
        plt.text(0, 0, f'TP = {phase_result_val["TP"]}', ha='center', va='bottom')
        plt.text(1, 0, f'FP = {phase_result_val["FP"]}', ha='center', va='bottom')
        plt.text(0, 1, f'FN = {phase_result_val["FN"]}', ha='center', va='bottom')
        plt.text(1, 1, f'TN = {phase_result_val["TN"]}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_fold, f'{TRAINING_METHOD}_predictions_phase_val-0717-644-{i}.jpg'), dpi = 500)


    # elif "content" in item:  # 成分生成任务
        

    else:#性能回归任务
        property_result_train = property_evaluate(data_pred_train, data_true_train, item)
        property_result_val = property_evaluate(data_pred_val, data_true_val, item)

        prediction_train = [float(value) for value in data_pred_train[item]]
        label_train = [float(value) for value in data_true_train[item]]
        prediction_val = [float(value) for value in data_pred_val[item]]
        label_val = [float(value) for value in data_true_val[item]]

        print(f"{item}_train R2: {property_result_train["r2"]} MAE: {property_result_train["mae"]} MSE: {property_result_train["mse"]} RMSE: {property_result_train["rmse"]}")
        print(f"{item}_val R2: {property_result_val["r2"]} MAE: {property_result_val["mae"]} MSE: {property_result_val["mse"]} RMSE: {property_result_val["rmse"]}")
       
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                            left=0.1, right=0.95, bottom=0.1, top=0.95,
                            wspace=0.05, hspace=0.05)

        # Main scatter plot
        ax = fig.add_subplot(gs[1, 0])
        min_value = min(min(label_train), min(prediction_train), min(label_val), min(prediction_val))
        max_value = max(max(label_train), max(prediction_train), max(label_val), max(prediction_val))

        # Plot training and validation data
        train_scatter = ax.scatter(label_train, prediction_train, color='blue', 
                                label=f'Trainset (R2={property_result_train["r2"]:.4f} MAE={property_result_train["mae"]:.4f})')
        val_scatter = ax.scatter(label_val, prediction_val, color='red', 
                                label=f'Valset (R2={property_result_val["r2"]:.4f} MAE={property_result_val["mae"]:.4f}')

        # Perfect prediction line
        ax.plot([min_value, max_value], [min_value, max_value], 'k--', lw=1)
        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predictions', fontsize=12)
        # ax.set_title(f'{item} Prediction Comparison', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)

        # Add histograms for marginal distributions
        # Top histogram (true values distribution)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_top.hist(label_train, bins=30, color='blue', alpha=0.5, )
        ax_top.hist(label_val, bins=30, color='red', alpha=0.5)
        ax_top.set_ylabel('Count')
        ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Right histogram (predictions distribution)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_right.hist(prediction_train, bins=30, color='blue', alpha=0.5, orientation='horizontal')
        ax_right.hist(prediction_val, bins=30, color='red', alpha=0.5, orientation='horizontal')
        ax_right.set_xlabel('Count')
        ax_right.tick_params(axis='y', which='both', left=False, labelleft=False)

        plt.savefig(os.path.join(output_fold, f'{TRAINING_METHOD}_predictions_with_marginals-0717-644-{i}.jpg'), dpi=500)
        plt.close()