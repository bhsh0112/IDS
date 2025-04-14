import torch
import numpy as np
from torch.utils.data import DataLoader
from lstm import LSTMModel, load_data
from tqdm import tqdm
from sklearn.metrics import f1_score

def test_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing")
        for inputs, labels in test_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds)
    return accuracy, f1

if __name__ == "__main__":
    train_path = "./data/UNSW_NB15/UNSW_NB15_training-set.csv"
    test_path = "./data/UNSW_NB15/UNSW_NB15_testing-set.csv"
    batch_size = 2048
    hidden_size = 128
    num_layers = 2
    dropout = 0.5

    _, test_dataset = load_data(train_path, test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    input_size = test_dataset.features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers, 1, dropout).to(device)

    model.load_state_dict(torch.load("./lstm/lstm_best_model.pth", map_location=device))
    print("Loaded best model weights.")

    accuracy, f1 = test_model(test_loader, model, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1-Score: {f1:.4f}")