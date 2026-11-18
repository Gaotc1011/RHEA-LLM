import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Simple Feedforward Neural Network Model
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
    
def load_and_preprocess_data(file_path, input_columns, task_col):
    df = pd.read_csv(file_path, encoding='utf-8')

    # Filter out rows where the target is 'N' or NaN
    valid_df = df[(df[task_col].notna()) & (df[task_col] != 'N')]

    X = np.nan_to_num(valid_df[input_columns].values, nan=0.0)
    y = valid_df[task_col].astype(float).values

    # Normalize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)#20 for YS1073，50 for YS1273/, 49 for hardness, 40 for strain, 43 for YS, other 42

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01, model_name="model"):#lr 0.01+100 for YS1073/YS1273,0.01 for YS/hardness other 0.001
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            test_loss = criterion(outputs, y_test)
            test_losses.append(test_loss.item())

        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            best_model_state = model.state_dict()
            train_outputs = model(X_train)
            test_outputs = model(X_test)
            print(f"Model {model_name} saved.")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {test_loss.item():.4f}, Best Loss: {best_loss:.4f}")

    # Save best model
    # best_model_state = model.state_dict()
    torch.save(best_model_state, f"./results/{model_name}_best.pth")
    train_outputs = model(X_train)
    test_outputs = model(X_test)

    return train_losses, test_losses, train_outputs, test_outputs

def plot_loss_curves(train_losses, test_losses, task):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and Test Loss Curve - {task}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{task}_loss_curve.png")
    plt.show()


def plot_true_vs_pred(y_train_true, y_train_pred, y_test_true, y_test_pred, task_name):
    y_train_true = y_train_true.detach().numpy().flatten()
    y_test_true = y_test_true.detach().numpy().flatten()
    y_train_pred = y_train_pred.detach().numpy().flatten()
    y_test_pred = y_test_pred.detach().numpy().flatten()

    # print(y_test_pred)
    # calculate metrics
    train_r2 = r2_score(y_train_true, y_train_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    train_mse = mean_squared_error(y_train_true, y_train_pred)
    test_mse = mean_squared_error(y_test_true, y_test_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_train_true, y_train_pred, alpha=0.7, label=f'Train Set R2 {train_r2:.4f} MAE {train_mae:.4f} MSE {train_mse:.4f} RMSE {train_rmse:.4f}', color='blue')
    plt.scatter(y_test_true, y_test_pred, alpha=0.7, label=f'Test Set R2 {test_r2:.4f} MAE {test_mae:.4f} MSE {test_mse:.4f} RMSE {test_rmse:.4f}', color='orange')

    all_true = np.concatenate([y_train_true, y_test_true])
    all_pred = np.concatenate([y_train_pred, y_test_pred])

    min_val = min(min(all_true), min(all_pred))
    max_val = max(max(all_true), max(all_pred))

    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted - {task_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./results/{task_name}_true_vs_pred_combined.png")
    plt.show()

def save_predictions_to_csv(y_train_true, y_train_pred, y_test_true, y_test_pred, task_name):
    # Flatten and convert tensors to numpy
    max_len = len(y_train_true)
    def pad_array(arr, length):
        return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)

    data = {
        'Train_True': y_train_true.detach().numpy().flatten(),
        'Train_Predicted': y_train_pred.detach().numpy().flatten(),
        'Test_True': pad_array(y_test_true.detach().numpy().flatten(), max_len),
        'Test_Predicted': pad_array(y_test_pred.detach().numpy().flatten(), max_len)
    }

    df = pd.DataFrame(data)
    df.to_csv(f"./results/{task_name}_predictions.csv", index=False)

def main():
    file_path = "./data/alloy_composition_data.csv"
    elements = ["Al","Cr","Hf","Mo","Nb","Si","Ta","Ti","V","W","Zr"]
    # tasks = ["density", "hardness", "compressive_yield_strength_at_room_temperature", "compressive_strain_at_room_temperature",
    #     "compressive_yield_strength_at_1073K",
    #     "compressive_yield_strength_at_1273K"  ]
    tasks = ["compressive_yield_strength_at_1273K"]
    

    for task in tasks:
        print(f"\nTraining model for task: {task}")
        try:
            X_train, y_train, X_test, y_test = load_and_preprocess_data(file_path, elements, task)
        except KeyError:
            print(f"Task '{task}' not found in dataset. Skipping.")
            continue

        # print(len(y_test))
        print(task)
        # exit(0)
        model = RegressionModel(len(elements))
        train_losses, test_losses, train_preds, test_preds = train_model(
            model, X_train, y_train, X_test, y_test, model_name=task
        )

        # Plot loss curves
        plot_loss_curves(train_losses, test_losses, task)

        # Plot true vs predicted
        plot_true_vs_pred(y_train, train_preds, y_test, test_preds, task)
        save_predictions_to_csv(y_train, train_preds, y_test, test_preds, task)

        print(f"Model for '{task}' finished training.")

if __name__ == "__main__":
    main()